# SPDX-License-Identifier: Apache-2.0
"""Dataset over the ``.pt`` blobs written by ``build_gen3c_cache.py``.

Each blob holds everything the diffusion loss needs, already encoded:

    video_latent          (16,16,88,160) fp16   tokenizer.encode(gt) * sigma_data
    condition_pose_latent (64,16,88,160) fp16   encoded warped frames + masks, buffer 2
    t5_text_embeddings    (512,1024)     fp16
    t5_text_mask          (512,)         int64
    meta                  dict           height/width/num_frames/sigma_data/...

The conditioner (``VideoConditionerFpsSizePadding``) additionally reads ``fps``, ``image_size``
and ``padding_mask``. ``image_size`` and ``padding_mask`` come from ``meta``. ``fps`` is not
recoverable from the cache, so it is synthesized as a constant that must match the value used at
inference (``gen3c_single_image.py`` and ``sample_from_cache.py`` both default to 24). Being
constant, it contributes a fixed bias to the timestep embedding rather than a real signal.
"""

import json
import os
from typing import Optional

import torch
from torch.utils.data import Dataset

from cosmos_predict1.utils import log

DEFAULT_FPS = 24


class Gen3CCacheDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
        fps: int = DEFAULT_FPS,
        expect_sigma_data: Optional[float] = 0.5,
        expect_frame_buffer_max: Optional[int] = 2,
    ):
        self.cache_dir = cache_dir
        self.fps = fps

        manifest_path = os.path.join(cache_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                sample_ids = json.load(f)["sample_ids"]
            self.paths = [os.path.join(cache_dir, f"{sid}.pt") for sid in sample_ids]
            self.paths = [p for p in self.paths if os.path.exists(p)]
        else:
            self.paths = sorted(
                os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".pt")
            )
        if not self.paths:
            raise FileNotFoundError(f"no cached .pt clips under {cache_dir}")

        # The cache is only interchangeable with the model if the VAE scaling and the warp buffer
        # width agree, and both are baked into the tensors rather than recoverable from them.
        probe = torch.load(self.paths[0], map_location="cpu", weights_only=False)["meta"]
        if expect_sigma_data is not None and probe["sigma_data"] != expect_sigma_data:
            raise ValueError(
                f"cache built with sigma_data={probe['sigma_data']}, model uses {expect_sigma_data}"
            )
        if expect_frame_buffer_max is not None and probe["frame_buffer_max"] != expect_frame_buffer_max:
            raise ValueError(
                f"cache built with frame_buffer_max={probe['frame_buffer_max']}, "
                f"model uses {expect_frame_buffer_max}"
            )
        log.info(
            f"[Gen3CCacheDataset] {len(self.paths)} clips from {cache_dir} "
            f"(geometry_source={probe.get('geometry_source', 'unknown')}, fps={fps})"
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        blob = torch.load(self.paths[index], map_location="cpu", weights_only=False)
        meta = blob["meta"]
        h, w, t = meta["height"], meta["width"], meta["num_frames"]

        return {
            "video_latent": blob["video_latent"].float(),
            "condition_pose_latent": blob["condition_pose_latent"].float(),
            "t5_text_embeddings": blob["t5_text_embeddings"].float(),
            "t5_text_mask": blob["t5_text_mask"],
            "fps": torch.tensor(self.fps, dtype=torch.float32),
            "num_frames": torch.tensor(t, dtype=torch.float32),
            "image_size": torch.tensor([h, w, h, w], dtype=torch.float32),
            "padding_mask": torch.zeros(1, h, w, dtype=torch.float32),
        }
