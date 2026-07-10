# SPDX-License-Identifier: Apache-2.0
"""
Sample the pretrained GEN3C model directly on a cached RealEstate10K clip (`<sample_id>.pt`).

The cache already stores the *encoded* conditioning, so this bypasses MoGe depth prediction,
Cache3D_Buffer warping, and the trajectory generator that `gen3c_single_image.py` runs:

    condition_pose_latent (64,16,88,160)  == DiffusionGen3CModel.encode_warped_frames(...)
    video_latent          (16,16,88,160)  == model.encode(gt_video); latent frame 0 is the
                                             image condition (num_input_frames=1)

Writes an mp4 with the model sample on top and the decoded GT on the bottom, so a pretrained
(zero-shot) run can be compared against the clip it was conditioned on.

    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python \
        cosmos_predict1/diffusion/training/datasets/realestate10k/sample_from_cache.py \
        --cache-dir datasets/gen3c_re10k/refined_train_10pct \
        --checkpoint-dir checkpoints \
        --out-dir datasets/gen3c_re10k/_samples --num 1
"""

import argparse
import glob
import os

import imageio
import numpy as np
import torch

from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.utils import log

NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special "
    "effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
    "flickering. Overall, the video is of poor quality."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True, help="A <set_name> dir holding <sample_id>.pt files.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--diffusion-transformer-dir", default="GEN3C-Cosmos-7B")
    p.add_argument("--num", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=35)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    p.add_argument("--prompt", default=None, help="Override the caption stored in the .pt.")
    p.add_argument("--no-gt", action="store_true", help="Skip decoding the GT latent for comparison.")
    p.add_argument("--offload-tokenizer", action="store_true")
    p.add_argument("--offload-network", action="store_true")
    p.add_argument("--offload-text-encoder-model", action="store_true")
    return p.parse_args()


@torch.no_grad()
def sample_one(pipeline: Gen3cPipeline, blob: dict, args) -> np.ndarray:
    """Run the diffusion sampler using the cached latents. Returns (T,H,W,3) uint8."""
    device = torch.device("cuda")
    dtype = pipeline.model.tensor_kwargs["dtype"]

    prompt = args.prompt or blob["meta"]["prompt"]
    prompts = [prompt, args.negative_prompt]
    prompt_embeddings, _ = pipeline._run_text_embedding_on_prompt_with_offload(prompts)

    # Frame-0 image condition. Only the first num_condition_t latent frames of this tensor are
    # read (the rest are masked out by condition_video_indicator), so handing over the full GT
    # latent is equivalent to encoding [frame0, zeros...] as get_condition_latent would.
    condition_latent = blob["video_latent"].unsqueeze(0).to(device, torch.bfloat16)

    # Substitute the cached warped-frame condition for the on-the-fly encode of Cache3D renders.
    latent_condition = blob["condition_pose_latent"].unsqueeze(0).to(device, dtype)
    pipeline.model.encode_warped_frames = lambda *_a, **_k: latent_condition

    if pipeline.offload_network:
        pipeline._load_network()
    sample = pipeline._run_model(
        embedding=prompt_embeddings[0],
        condition_latent=condition_latent,
        rendered_warp_images=None,  # consumed only by the patched encode_warped_frames
        rendered_warp_masks=None,
        negative_prompt_embedding=prompt_embeddings[1],
    )
    if pipeline.offload_network:
        pipeline._offload_network()

    if pipeline.offload_tokenizer:
        pipeline._load_tokenizer()
    video = pipeline._run_tokenizer_decoding(sample)  # (T,H,W,3) uint8
    if pipeline.offload_tokenizer:
        pipeline._offload_tokenizer()
    return video


@torch.no_grad()
def decode_gt(pipeline: Gen3cPipeline, blob: dict) -> np.ndarray:
    if pipeline.offload_tokenizer:
        pipeline._load_tokenizer()
    latent = blob["video_latent"].unsqueeze(0).to("cuda", torch.bfloat16)
    pixels = pipeline.model.decode(latent)[0]  # (3,T,H,W) in ~[-1,1]
    if pipeline.offload_tokenizer:
        pipeline._offload_tokenizer()
    pixels = ((pixels.float().clamp(-1, 1) * 0.5 + 0.5) * 255).to(torch.uint8)
    return pixels.permute(1, 2, 3, 0).cpu().numpy()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pipeline = Gen3cPipeline(
        inference_type="video2world",
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.diffusion_transformer_dir,
        enable_prompt_upsampler=False,
        offload_network=args.offload_network,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_prompt_upsampler=True,
        offload_guardrail_models=True,
        disable_guardrail=True,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=704,
        width=1280,
        fps=args.fps,
        num_video_frames=121,
        seed=args.seed,
    )

    files = sorted(glob.glob(os.path.join(args.cache_dir, "*.pt")))[: args.num]
    if not files:
        raise SystemExit(f"no .pt files under {args.cache_dir}")

    for path in files:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        sample_id = blob["meta"]["sample_id"]
        assert blob["meta"]["sigma_data"] == pipeline.model.sigma_data, (
            f"cache built with sigma_data={blob['meta']['sigma_data']}, "
            f"model uses {pipeline.model.sigma_data}"
        )
        log.info(f"sampling {sample_id}: {blob['meta']['prompt'][:70]!r}")

        video = sample_one(pipeline, blob, args)
        if not args.no_gt:
            video = np.concatenate([video, decode_gt(pipeline, blob)], axis=1)

        out = os.path.join(args.out_dir, f"{sample_id}_sample.mp4")
        writer = imageio.get_writer(out, fps=args.fps, quality=5)
        for frame in video:
            writer.append_data(frame)
        writer.close()
        print(f"wrote {out}  {video.shape}")


if __name__ == "__main__":
    main()
