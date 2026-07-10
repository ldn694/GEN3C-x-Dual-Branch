# SPDX-License-Identifier: Apache-2.0
"""Training-side counterpart of ``diffusion/model/model_gen3c.DiffusionGen3CModel``.

GEN3C conditions the DiT on the VAE-encoded renders of a 3D point-cloud cache warped into
each target view, concatenated along the channel dim as
``(warped_frames + warped_frames_mask) * frame_buffer_max`` = 64 channels for buffer 2.

``ExtendDiffusionModel`` already carries the whole conditioning machinery -- the
``condition_video_pose`` field, the network concat in ``general_dit_lvg.VideoExtendGeneralDIT``,
and the condition-region composition in ``denoise()``. Only two things differ:

1. ``add_condition_pose`` reads ``plucker_embeddings`` (the Cosmos camera-control variant)
   rather than the encoded warp buffer.
2. ``get_data_and_condition`` runs the VAE on raw pixels. Our dataloader serves latents that
   ``build_gen3c_cache.py`` already encoded, so the encode must be skipped.

Inheriting ``denoise()`` unchanged is the point of this file: it guarantees the training-time
condition composition matches ``DiffusionGen3CModel`` at inference.
"""

from typing import Dict, Tuple, Union

import torch
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict1.diffusion.training.conditioner import DataType, VideoExtendCondition
from cosmos_predict1.diffusion.training.models.extend_model import ExtendDiffusionModel
from cosmos_predict1.diffusion.training.models.model import _broadcast, broadcast_condition
from cosmos_predict1.diffusion.training.models.model_image import diffusion_fsdp_class_decorator
from cosmos_predict1.diffusion.training.models.model_peft import video_peft_decorator
from cosmos_predict1.utils import log

CONDITION_POSE_LATENT_KEY = "condition_pose_latent"


class Gen3CDiffusionModel(ExtendDiffusionModel):
    def __init__(self, config):
        super().__init__(config)
        self.frame_buffer_max = config.frame_buffer_max

    def get_data_and_condition(
        self, data_batch: dict[str, Tensor], num_condition_t: Union[int, None] = None
    ) -> Tuple[Tensor, Tensor, VideoExtendCondition]:
        """Latent-in variant of ``DiffusionModel.get_data_and_condition``.

        The cache stores ``tokenizer.encode(gt) * sigma_data``, which is exactly what
        ``self.encode()`` would return, so ``latent_state`` is used directly. ``raw_state``
        has no pixel counterpart here; the loss path only reads its ``.shape``, so the latent
        stands in for it.
        """
        assert not self.is_image_batch(data_batch), "Gen3C training is video-only."

        # sort keys to make sure the order is same across ranks, IMPORTANT! otherwise, nccl will hang!
        for key in sorted(list(data_batch.keys())):
            data_batch[key] = _broadcast(data_batch[key], to_tp=True, to_cp=True)

        latent_state = data_batch[self.input_data_key].to(**self.tensor_kwargs).contiguous()

        condition = self.conditioner(data_batch)
        condition.data_type = DataType.VIDEO

        latent_state = _broadcast(latent_state, to_tp=True, to_cp=True)
        condition = broadcast_condition(condition, to_tp=True, to_cp=True)

        if self.config.conditioner.video_cond_bool.sample_tokens_start_from_p_or_i:
            latent_state = self.sample_tokens_start_from_p_or_i(latent_state)
        condition = self.add_condition_video_indicator_and_video_input_mask(
            latent_state, condition, num_condition_t=num_condition_t
        )
        assert self.config.conditioner.video_cond_bool.add_pose_condition, (
            "Gen3C requires add_pose_condition=True; without it the network gets "
            f"{self.net.in_channels - 64} of its {self.net.in_channels} input channels."
        )
        condition = self.add_condition_pose(data_batch, condition)

        log.debug(f"condition.data_type {condition.data_type}")
        return latent_state, latent_state, condition

    def add_condition_pose(self, data_batch: Dict, condition: VideoExtendCondition) -> VideoExtendCondition:
        """Attach the cached warped-frame latent as the pose condition.

        Mirrors ``DiffusionGen3CModel.add_condition_pose``: under classifier-free-guidance
        dropout (``video_cond_bool`` False) the pose latent is zeroed, matching the
        ``drop_out_latent=True`` branch the inference model uses for its uncondition.
        """
        assert CONDITION_POSE_LATENT_KEY in data_batch, (
            f"{CONDITION_POSE_LATENT_KEY!r} should be in data_batch. only find {list(data_batch.keys())}"
        )
        latent_condition = data_batch[CONDITION_POSE_LATENT_KEY].to(**self.tensor_kwargs)

        assert condition.video_cond_bool is not None, "video_cond_bool should be set"
        if condition.video_cond_bool:
            condition.condition_video_pose = latent_condition.contiguous()
        else:
            condition.condition_video_pose = torch.zeros_like(latent_condition).contiguous()

        to_cp = self.net.is_context_parallel_enabled
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
        else:
            assert not to_cp, "parallel_state is not initialized, context parallel should be turned off."

        return condition


@diffusion_fsdp_class_decorator
class FSDPGen3CDiffusionModel(Gen3CDiffusionModel):
    pass


@video_peft_decorator
class PEFTGen3CDiffusionModel(Gen3CDiffusionModel):
    pass
