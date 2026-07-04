# SPDX-License-Identifier: Apache-2.0
"""
De-risk the GEN3C cache before a mass run: decode a cached latent back to pixels and eyeball it.

For each sample it writes an mp4 with the reconstructed GT video (top) and the buffer-0 warped
condition video (bottom). If the top looks like the clip and the bottom looks like the clip
forward-warped along the trajectory (holes where geometry is disoccluded), the cache is sound.

    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python \
        cosmos_predict1/diffusion/training/datasets/realestate10k/verify_cache.py \
        --cache-dir datasets/gen3c_re10k/refined_train_10pct \
        --checkpoint-dir checkpoints \
        --out-dir datasets/gen3c_re10k/_verify --num 3
"""

import argparse
import glob
import os

import imageio
import numpy as np
import torch

from cosmos_predict1.diffusion.training.datasets.realestate10k.build_gen3c_cache import load_tokenizer


@torch.no_grad()
def decode_latent(tokenizer, latent: torch.Tensor, sigma_data: float, device: str) -> np.ndarray:
    """latent (C,T,H,W) fp16 -> (T,H,W,3) uint8. Inverts tokenizer.encode()*sigma_data."""
    latent = latent.unsqueeze(0).to(device).to(tokenizer.video_vae.dtype) / sigma_data
    pixels = tokenizer.decode(latent)[0]                      # (3,T,H,W) in ~[-1,1]
    pixels = ((pixels.float().clamp(-1, 1) * 0.5 + 0.5) * 255).to(torch.uint8)
    return pixels.permute(1, 2, 3, 0).cpu().numpy()          # (T,H,W,3)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True, help="A <set_name> dir holding <sample_id>.pt files.")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--tokenizer-subdir", default="Cosmos-Tokenize1-CV8x8x8-720p")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num", type=int, default=3)
    p.add_argument("--fps", type=int, default=24)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = load_tokenizer(os.path.join(args.checkpoint_dir, args.tokenizer_subdir), device)

    files = sorted(glob.glob(os.path.join(args.cache_dir, "*.pt")))[: args.num]
    for path in files:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        sigma_data = blob["meta"]["sigma_data"]
        gt = decode_latent(tokenizer, blob["video_latent"], sigma_data, device)          # (T,H,W,3)
        warp0 = decode_latent(tokenizer, blob["condition_pose_latent"][:16], sigma_data, device)
        stacked = np.concatenate([gt, warp0], axis=1)                                    # stack vertically
        out = os.path.join(args.out_dir, blob["meta"]["sample_id"] + "_verify.mp4")
        writer = imageio.get_writer(out, fps=args.fps, quality=5)
        for frame in stacked:
            writer.append_data(frame)
        writer.close()
        print(f"wrote {out}  gt={gt.shape} warp0={warp0.shape}  prompt={blob['meta']['prompt'][:60]!r}")


if __name__ == "__main__":
    main()
