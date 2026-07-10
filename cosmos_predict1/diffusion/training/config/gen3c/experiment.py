# SPDX-License-Identifier: Apache-2.0
"""LoRA finetuning of GEN3C-Cosmos-7B on the cached RealEstate10K clips.

Run with:

    torchrun --nproc_per_node=4 -m cosmos_predict1.diffusion.training.train \
        --config=cosmos_predict1/diffusion/training/config/config.py \
        -- experiment=gen3c_7b_lora_realestate10k
"""

from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict1.diffusion.training.callbacks.iter_speed import IterSpeed
from cosmos_predict1.diffusion.training.callbacks.low_precision import LowPrecisionCallback
from cosmos_predict1.diffusion.training.datasets.realestate10k.gen3c_cache_dataset import Gen3CCacheDataset
from cosmos_predict1.diffusion.training.models.model_gen3c import PEFTGen3CDiffusionModel
from cosmos_predict1.diffusion.training.networks.general_dit_lvg import VideoExtendGeneralDIT
from cosmos_predict1.diffusion.training.utils.peft.lora_config import get_fa_ca_qv_lora_config
from cosmos_predict1.utils import log
from cosmos_predict1.utils.callback import ProgressBarCallback
from cosmos_predict1.utils.callbacks.grad_clip import GradClip
from cosmos_predict1.utils.lazy_config import PLACEHOLDER
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict

FRAME_BUFFER_MAX = 2
NUM_FRAMES = 121
# 16 video latent + 64 encoded warp buffer (2 views x (rgb + mask) x 16) + 1 condition-region mask
IN_CHANNELS = 16 + 16 * 2 * FRAME_BUFFER_MAX + 1


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


gen3c_cache_dataset_train = L(Gen3CCacheDataset)(
    cache_dir="datasets/gen3c_re10k/refined_train_10pct",
    fps=24,
    expect_frame_buffer_max=FRAME_BUFFER_MAX,
)

dataloader_train_gen3c_re10k = L(DataLoader)(
    dataset=gen3c_cache_dataset_train,
    sampler=L(get_sampler)(dataset=gen3c_cache_dataset_train),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=4,
)

gen3c_7b_lora_realestate10k = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "peft"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="gen3c",
            name="gen3c_7b_lora_realestate10k",
        ),
        optimizer=dict(
            lr=1e-4,
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=1000,
            broadcast_via_filesystem=True,
            load_path="checkpoints/Gen3C-Cosmos-7B/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,
        ),
        trainer=dict(
            max_iter=5000,
            distributed_parallelism="ddp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=False,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=4,
        ),
        model=dict(
            peft_control=get_fa_ca_qv_lora_config(first_nblocks=28, rank=8, scale=1),
            # The dataloader serves latents, not pixels; this key selects them in
            # Gen3CDiffusionModel.get_data_and_condition and drives is_image_batch().
            input_data_key="video_latent",
            latent_shape=[16, 16, 88, 160],
            frame_buffer_max=FRAME_BUFFER_MAX,
            loss_reduce="mean",
            ema=dict(enabled=False),
            fsdp_enabled=False,
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
                in_channels=IN_CHANNELS,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    add_pose_condition=True,
                    # Inference conditions on latent frame 0 only (num_input_frames=1); pinning
                    # min == max == 1 makes training see exactly that condition region.
                    condition_location="first_random_n",
                    first_random_n_num_condition_t_min=1,
                    first_random_n_num_condition_t_max=1,
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,
                    normalize_condition_latent=False,
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(pixel_chunk_duration=NUM_FRAMES),
        ),
        model_obj=L(PEFTGen3CDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        scheduler=dict(
            warm_up_steps=[0],
        ),
        dataloader_train=dataloader_train_gen3c_re10k,
        dataloader_val=dataloader_train_gen3c_re10k,
    )
)


def register_experiments(cs):
    for _item in [
        gen3c_7b_lora_realestate10k,
    ]:
        experiment_name = _item["job"]["name"]
        log.info(f"Registering experiment: {experiment_name}")
        cs.store(group="experiment", package="_global_", name=experiment_name, node=_item)
