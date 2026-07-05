# SPDX-License-Identifier: Apache-2.0
"""
Offline builder: RealEstate10K clip -> GEN3C training-ready latent cache.

For each usable clip it writes one file ``<output_dir>/<set_name>/<sample_id>.pt`` containing:

    video_latent          : (16, 16, 88, 160) fp16   target video latent = tokenizer.encode(gt)*sigma_data
    condition_pose_latent : (64, 16, 88, 160) fp16   GEN3C warped-frame condition (encode_warped_frames),
                                                      N=1 real source buffer + 1 zero-padded buffer
    t5_text_embeddings    : (L, 1024) fp16
    t5_text_mask          : (L,) int64
    meta                  : dict (sample_id, prompt, shapes, config)

Run on the GPU server (the Cosmos tokenizer + 3D warp both need CUDA):

    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python \
        cosmos_predict1/diffusion/training/datasets/realestate10k/build_gen3c_cache.py \
        --dataset-root /path/to/realestate10k \
        --set-name refined_train_10pct \
        --checkpoint-dir checkpoints \
        --output-dir datasets/gen3c_re10k

With VGGT-Omega geometry inference (instead of preprocessed Voyager cameras/depth):

    python cosmos_predict1/diffusion/training/datasets/realestate10k/build_gen3c_cache.py \
        --dataset-root /path/to/realestate10k \
        --set-name refined_train_10pct \
        --checkpoint-dir checkpoints \
        --output-dir datasets/gen3c_re10k \
        --rerun-vggt \
        --vggt-checkpoint /path/to/vggt_omega_1b_512.pt \
        --vggt-image-resolution 512

Notes
-----
* video_latent doubles as the video2world condition source at train time: the training model sets
  condition_latent = gt_latent and masks the first ``num_condition_t`` latent frames. So NO separate
  first-frame condition tensor is cached.
* Matches GEN3C inference defaults for the warp: frame_buffer_max=2, filter_points_threshold=0.05,
  noise_aug_strength=0.0, is_depth=True. align_depth is NOT triggered (only used in update_cache).
* When --rerun-vggt is set, the first 121 frames of video are loaded from RealEstate10K raw data,
  sent through VGGT-Omega to infer camera poses and depth (resizing back to target 704×1280 resolution),
  then used for 3D warping instead of the preprocessed Voyager cameras/depth.
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from cosmos_predict1.diffusion.config.base.tokenizer import get_cosmos_diffusion_tokenizer_comp8x8x8
from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Buffer
from cosmos_predict1.diffusion.training.datasets.realestate10k.realestate_gen3c_dataset import (
    RealEstate10KGen3C,
)
from cosmos_predict1.utils.lazy_config import instantiate as lazy_instantiate

torch.enable_grad(False)

FRAME_BUFFER_MAX = 2
FILTER_POINTS_THRESHOLD = 0.05
NOISE_AUG_STRENGTH = 0.0


# --------------------------------------------------------------------------------------
# Tokenizer (standalone; mirrors DiffusionT2WModel.set_up_tokenizer + .encode)
# --------------------------------------------------------------------------------------
def load_tokenizer(vae_dir: str, device: str):
    tok_cfg = get_cosmos_diffusion_tokenizer_comp8x8x8(resolution="720", chunk_duration=121)
    tokenizer = lazy_instantiate(tok_cfg)
    tokenizer.load_weights(vae_dir)
    tokenizer.reset_dtype()
    return tokenizer.to(device).eval()


@torch.no_grad()
def tok_encode(tokenizer, x: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """x: (B,3,T,H,W) in [-1,1], T divisible by 121 -> (B,16,T//8+?,H/8,W/8) scaled by sigma_data."""
    return tokenizer.encode(x) * sigma_data


@torch.no_grad()
def encode_warped_frames(tokenizer, condition_state, condition_state_mask, sigma_data, dtype):
    """Replicates DiffusionGen3CModel.encode_warped_frames (model_gen3c.py).

    condition_state:      (B,F,N,3,H,W) in [-1,1]
    condition_state_mask: (B,F,N,1,H,W) in {0,1}
    returns:              (B, (16+16)*FRAME_BUFFER_MAX, T_lat, H/8, W/8)
    """
    condition_state_mask = (condition_state_mask * 2 - 1).repeat(1, 1, 1, 3, 1, 1)
    latent_condition = []
    for i in range(condition_state.shape[2]):
        v = tok_encode(tokenizer, condition_state[:, :, i].permute(0, 2, 1, 3, 4).to(dtype), sigma_data).contiguous()
        m = tok_encode(tokenizer, condition_state_mask[:, :, i].permute(0, 2, 1, 3, 4).to(dtype), sigma_data).contiguous()
        latent_condition.append(v)
        latent_condition.append(m)
    for _ in range(FRAME_BUFFER_MAX - condition_state.shape[2]):
        latent_condition.append(torch.zeros_like(v))
        latent_condition.append(torch.zeros_like(m))
    return torch.cat(latent_condition, dim=1)


# --------------------------------------------------------------------------------------
# T5-XXL text embeddings (mirrors scripts/get_t5_embeddings.py)
# --------------------------------------------------------------------------------------
def init_t5(model_name: str, cache_dir: str, max_length: int, device: str):
    from transformers import T5EncoderModel, T5TokenizerFast

    tokenizer = T5TokenizerFast.from_pretrained(model_name, model_max_length=max_length, cache_dir=cache_dir)
    encoder = T5EncoderModel.from_pretrained(model_name, cache_dir=cache_dir).to(device).eval()
    return tokenizer, encoder


@torch.inference_mode()
def encode_t5(tokenizer, encoder, prompt: str, max_length: int, device: str):
    enc = tokenizer.batch_encode_plus(
        [prompt], return_tensors="pt", truncation=True, padding="max_length",
        max_length=max_length, return_length=True, return_offsets_mapping=False,
    )
    input_ids = enc.input_ids.to(device)
    attn_mask = enc.attention_mask.to(device)
    out = encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
    length = int(attn_mask.sum(dim=1)[0].item())
    out[0, length:] = 0
    emb = out[0, :length].cpu().to(torch.float16)          # (L, 1024)
    mask = torch.ones(length, dtype=torch.int64)
    return emb, mask


# --------------------------------------------------------------------------------------
# VGGT-Omega geometry inference
# --------------------------------------------------------------------------------------
def load_vggt_model(checkpoint_path: str, device: str):
    """Load VGGT-Omega model for camera and depth prediction."""
    try:
        from vggt_omega.models import VGGTOmega
    except ImportError:
        raise ImportError("vggt_omega is not installed. Install it via: pip install -e /path/to/vggt-omega")

    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device)


@torch.no_grad()
def run_vggt_inference(model, rgb_frames: torch.Tensor, image_resolution: int, device: str):
    """Run VGGT-Omega inference on RGB frames to get cameras and depth.

    Args:
        model: VGGT-Omega model
        rgb_frames: (T,3,H,W) uint8 RGB frames in [0,255]
        image_resolution: target resolution for vggt inference
        device: cuda or cpu

    Returns:
        intrinsics: (T,3,3) pixel intrinsics K
        w2c: (T,4,4) world-to-camera extrinsics (OpenCV convention)
        depth0: (1,1,H,W) metric depth for frame 0
    """
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    T, _, H, W = rgb_frames.shape

    # Prepare frames for vggt by converting to PIL images and loading via vggt utils
    import tempfile
    import os
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        for t in range(T):
            frame_pil = Image.fromarray(rgb_frames[t].permute(1, 2, 0).cpu().numpy().astype('uint8'))
            frame_path = os.path.join(tmpdir, f"{t:06d}.png")
            frame_pil.save(frame_path)
            image_paths.append(frame_path)

        images = load_and_preprocess_images(image_paths, image_resolution=image_resolution).to(device)

    _, _, vggt_H, vggt_W = images.shape

    predictions = model(images)

    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"],
        (vggt_H, vggt_W),
    )

    depth = predictions["depth"]  # (T,H_vggt,W_vggt,1)

    # Resize depth and intrinsics back to original frame resolution
    extrinsics_np = extrinsics.cpu().numpy()  # (T,3,4)
    intrinsics_np = intrinsics.cpu().numpy()  # (T,3,3)
    depth_np = depth.cpu().numpy()  # (T,H_vggt,W_vggt,1)

    # Scale intrinsics from vggt resolution to target resolution
    scale_x = W / vggt_W
    scale_y = H / vggt_H
    intrinsics_scaled = intrinsics_np.copy()
    intrinsics_scaled[:, 0, 0] *= scale_x  # fx
    intrinsics_scaled[:, 1, 1] *= scale_y  # fy
    intrinsics_scaled[:, 0, 2] = intrinsics_np[:, 0, 2] * scale_x  # cx
    intrinsics_scaled[:, 1, 2] = intrinsics_np[:, 1, 2] * scale_y  # cy

    # Resize depth to original resolution
    depth_resized = torch.nn.functional.interpolate(
        torch.from_numpy(depth_np).permute(0, 3, 1, 2).float(),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).numpy()  # (T,H,W,1)

    # Convert extrinsics (camera-from-world, OpenCV) to w2c (4x4)
    w2c_list = []
    for t in range(T):
        w2c_t = np.eye(4)
        w2c_t[:3, :3] = extrinsics_np[t, :3, :3]
        w2c_t[:3, 3] = extrinsics_np[t, :3, 3]
        w2c_list.append(w2c_t)

    intrinsics_torch = torch.from_numpy(intrinsics_scaled).float()  # (T,3,3)
    w2c_torch = torch.from_numpy(np.stack(w2c_list)).float()  # (T,4,4)
    depth0_torch = torch.from_numpy(depth_resized[0:1]).permute(0, 3, 1, 2).float()  # (1,1,H,W)

    return intrinsics_torch, w2c_torch, depth0_torch


# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build GEN3C latent cache from RealEstate10K.")
    p.add_argument("--dataset-root", required=True, help="RealEstate10K raw root (Voyager layout).")
    p.add_argument("--set-name", required=True, help="Split manifest folder, e.g. refined_train_10pct.")
    p.add_argument("--output-dir", required=True, help="Where to write <set_name>/<sample_id>.pt.")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--tokenizer-subdir", default="Cosmos-Tokenize1-CV8x8x8-720p")
    p.add_argument("--cameras-name", default="cameras")
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--sigma-data", type=float, default=0.5)
    p.add_argument("--t5-model-name", default="google-t5/t5-11b")
    p.add_argument("--t5-max-length", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--overwrite", action="store_true", help="Recompute even if the output exists.")
    p.add_argument("--limit", type=int, default=-1, help="Debug: only process the first N clips.")
    p.add_argument("--rerun-vggt", action="store_true", help="Infer depth/cameras from video using VGGT-Omega instead of preprocessed data.")
    p.add_argument("--vggt-checkpoint", default=None, help="Path to VGGT-Omega checkpoint (required if --rerun-vggt is set).")
    p.add_argument("--vggt-image-resolution", type=int, default=512, help="Image resolution for VGGT-Omega inference.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(args.output_dir, args.set_name)
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = load_tokenizer(os.path.join(args.checkpoint_dir, args.tokenizer_subdir), device)
    tok_dtype = tokenizer.video_vae.dtype
    t5_tokenizer, t5_encoder = init_t5(args.t5_model_name, args.checkpoint_dir, args.t5_max_length, device)

    vggt_model = None
    if args.rerun_vggt:
        if args.vggt_checkpoint is None:
            raise ValueError("--vggt-checkpoint must be specified when --rerun-vggt is set")
        if not os.path.exists(args.vggt_checkpoint):
            raise FileNotFoundError(f"VGGT checkpoint not found: {args.vggt_checkpoint}")
        print(f"Loading VGGT-Omega model from {args.vggt_checkpoint}")
        vggt_model = load_vggt_model(args.vggt_checkpoint, device)
        print("VGGT-Omega model loaded successfully")

    dataset = RealEstate10KGen3C(
        root_path=args.dataset_root, set_name=args.set_name, num_frames=args.num_frames,
        height=args.height, width=args.width, cameras_name=args.cameras_name,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, collate_fn=lambda b: b[0])

    manifest = []
    processed = 0
    for item in loader:
        sample_id = item["sample_id"]
        out_path = os.path.join(out_dir, f"{sample_id}.pt")
        if os.path.exists(out_path) and not args.overwrite:
            manifest.append(sample_id)
            continue

        # ---- target video latent -------------------------------------------------------
        rgb = item["rgb"].to(device).permute(1, 0, 2, 3).unsqueeze(0).float() / 127.5 - 1.0  # (1,3,T,H,W)
        video_latent = tok_encode(tokenizer, rgb, args.sigma_data)[0].cpu().to(torch.float16)

        # ---- GEN3C warped-frame condition (N=1 source buffer, frame 0) ------------------
        if args.rerun_vggt:
            # Infer depth and cameras from video using VGGT-Omega
            print(f"  [VGGT] Running inference on {sample_id} ({args.num_frames} frames)...")
            K, w2c, depth0 = run_vggt_inference(
                vggt_model, item["rgb"], args.vggt_image_resolution, device
            )
            K = K.to(device)
            w2c = w2c.to(device)
            depth0 = depth0.to(device)
        else:
            depth0 = item["depth0"].unsqueeze(0).to(device)                       # (1,1,H,W)
            K = item["intrinsic"].to(device)                                      # (T,3,3)
            w2c = item["w2c"].to(device)                                          # (T,4,4)

        cache = Cache3D_Buffer(
            frame_buffer_max=FRAME_BUFFER_MAX,
            generator=torch.Generator(device=device).manual_seed(0),
            noise_aug_strength=NOISE_AUG_STRENGTH,
            input_image=rgb[:, :, 0],                                         # (1,3,H,W) in [-1,1]
            input_depth=depth0,                                              # (1,1,H,W) metric z
            input_w2c=w2c[0:1],                                              # (1,4,4)
            input_intrinsics=K[0:1],                                         # (1,3,3)
            filter_points_threshold=FILTER_POINTS_THRESHOLD,
            is_depth=True,
            device=device,
        )
        warp_imgs, warp_masks = cache.render_cache(w2c.unsqueeze(0), K.unsqueeze(0))  # (1,T,N,3,H,W),(1,T,N,1,H,W)
        condition_pose_latent = encode_warped_frames(
            tokenizer, warp_imgs, warp_masks, args.sigma_data, tok_dtype)[0].cpu().to(torch.float16)

        # ---- text ----------------------------------------------------------------------
        t5_emb, t5_mask = encode_t5(t5_tokenizer, t5_encoder, item["prompt"], args.t5_max_length, device)

        torch.save({
            "video_latent": video_latent,                    # (16,16,88,160) fp16
            "condition_pose_latent": condition_pose_latent,  # (64,16,88,160) fp16
            "t5_text_embeddings": t5_emb,                    # (L,1024) fp16
            "t5_text_mask": t5_mask,                         # (L,) int64
            "meta": {
                "sample_id": sample_id,
                "prompt": item["prompt"],
                "video_latent_shape": tuple(video_latent.shape),
                "condition_pose_latent_shape": tuple(condition_pose_latent.shape),
                "num_frames": args.num_frames,
                "height": args.height,
                "width": args.width,
                "sigma_data": args.sigma_data,
                "frame_buffer_max": FRAME_BUFFER_MAX,
                "filter_points_threshold": FILTER_POINTS_THRESHOLD,
                "geometry_source": "vggt-omega" if args.rerun_vggt else "voyager",
            },
        }, out_path)
        manifest.append(sample_id)
        processed += 1
        geom_src = "vggt" if args.rerun_vggt else "voyager"
        print(f"[{processed}] wrote {out_path}  video={tuple(video_latent.shape)} "
              f"pose={tuple(condition_pose_latent.shape)} t5={tuple(t5_emb.shape)} geom={geom_src}")

        if args.limit > 0 and processed >= args.limit:
            break

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"set_name": args.set_name, "sample_ids": manifest}, f, indent=2)
    print(f"Done. {processed} newly built, {len(manifest)} total in manifest.")


if __name__ == "__main__":
    main()
