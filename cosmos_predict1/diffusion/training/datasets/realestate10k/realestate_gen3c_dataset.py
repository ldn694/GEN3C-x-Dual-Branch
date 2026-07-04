# SPDX-License-Identifier: Apache-2.0
"""
Raw RealEstate10K loader that yields exactly what the GEN3C data-prep builder needs.

This is adapted from the Voyager-x-FramePack loader (``dataset/RealEstate10K.py``) but with
three deliberate differences for the GEN3C (Cosmos-Predict1) pipeline:

1. Geometry maps the source (~1280x720) to the GEN3C target 704x1280 via *resize-to-cover +
   center-crop* (for exactly 1280x720 this reduces to cropping 8px off top and bottom, cy-=8,
   with no resampling of RGB or depth). Voyager's crop assumed full-height + horizontal crop,
   which asserts-out for the wider 704-tall aspect.
2. Depth is returned as **metric z** (1 / inverse_depth), which is what GEN3C's Cache3D wants
   with ``is_depth=True``. Only frame 0's depth is decoded (N=1 buffer strategy).
3. Frames are the first ``num_frames`` (default 121) at interval 1, starting at index 0, so the
   clip is a real contiguous camera trajectory.

Reuses Voyager's raw layout unchanged:
    <root>/transcode/<video_id>/<timestamp>.jpg
    <root>/depth_videos/<sample_id>.avi          (uint16 inverse-depth video)
    <root>/depth_ranges/<sample_id>.json         (per-frame min/max inverse depth)
    <root>/<cameras_name>/<sample_id>.json       (per-frame 3x3 intrinsic + 4x4 w2c)
    <root>/train_caption.json
    <root>/<set_name>/*.txt                       (split manifest, one clip per file)
"""

import glob
import json
import os

import av
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image
from torch.utils.data import Dataset


def compute_resize_crop(H0: int, W0: int, Ht: int, Wt: int):
    """Resize-to-cover the target then center-crop. Returns (scale, Hr, Wr, x0, y0).

    scale is chosen so the resized image covers (Ht, Wt); the excess is center-cropped.
    For H0,W0 = 720,1280 and Ht,Wt = 704,1280 this gives scale=1.0, x0=0, y0=8 (pure crop).
    """
    scale = max(Wt / W0, Ht / H0)
    Wr, Hr = round(W0 * scale), round(H0 * scale)
    x0 = (Wr - Wt) // 2
    y0 = (Hr - Ht) // 2
    return scale, Hr, Wr, x0, y0


class RealEstate10KGen3C(Dataset):
    def __init__(
        self,
        root_path: str,
        set_name: str,
        num_frames: int = 121,
        height: int = 704,
        width: int = 1280,
        cameras_name: str = "cameras",
    ):
        self.root_path = root_path
        self.set_name = set_name
        self.num_frames = num_frames
        self.height = height
        self.width = width

        self.frame_path = os.path.join(root_path, "transcode")
        self.depth_path = os.path.join(root_path, "depth_videos")
        self.depth_ranges_path = os.path.join(root_path, "depth_ranges")
        self.cameras_path = os.path.join(root_path, cameras_name)
        self.caption_path = os.path.join(root_path, "train_caption.json")
        self.folder_path = os.path.join(root_path, set_name)

        with open(self.caption_path, "r") as f:
            self.captions = json.load(f)

        self.data = []
        media_files = sorted(glob.glob(os.path.join(self.folder_path, "*.txt")))
        for media_file in tqdm.tqdm(media_files, desc=f"Indexing {set_name}"):
            timestamps = []
            with open(media_file) as f:
                video_id = f.readline().rstrip().split("=")[-1]
                for line in f.readlines():
                    timestamps.append(int(line.split(" ")[0]))
            if len(timestamps) < self.num_frames:
                continue

            sample_id = os.path.basename(media_file).split(".")[0]
            camera_file = os.path.join(self.cameras_path, sample_id + ".json")
            depth_video_file = os.path.join(self.depth_path, sample_id + ".avi")
            depth_range_file = os.path.join(self.depth_ranges_path, sample_id + ".json")
            rgb_folder = os.path.join(self.frame_path, video_id)
            if not (os.path.exists(camera_file) and os.path.exists(depth_video_file)
                    and os.path.exists(depth_range_file) and os.path.exists(rgb_folder)):
                continue

            with open(camera_file) as f:
                camera_data = json.load(f)
            if len(camera_data) < self.num_frames:
                continue

            prompt = self.captions.get(sample_id, "")
            prompt = prompt.split(";")[0].strip() if prompt else ""

            self.data.append({
                "sample_id": sample_id,
                "timestamps": timestamps,
                "camera_data": camera_data,
                "rgb_folder": rgb_folder,
                "depth_video_file": depth_video_file,
                "depth_range_file": depth_range_file,
                "prompt": prompt,
            })
        print(f"[RealEstate10KGen3C] {len(self.data)} usable clips in '{set_name}'.")

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _to_pixel_intrinsic(K: torch.Tensor, W0: int, H0: int) -> torch.Tensor:
        """RealEstate10K intrinsics may be stored normalized (fx as a fraction of width).
        Convert to pixels if so. Heuristic: normalized fx is << 1, pixel fx is > 1."""
        K = K.clone()
        if float(K[0, 0]) < 1.0:  # normalized -> pixels
            K[0, 0] *= W0
            K[0, 2] *= W0
            K[1, 1] *= H0
            K[1, 2] *= H0
        return K

    def _decode_depth_frame0_inverse(self, depth_video_file: str, lo: float, hi: float) -> np.ndarray:
        """Decode frame 0 of the uint16 depth video and de-quantize to inverse depth."""
        with av.open(depth_video_file) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                d_u16 = frame.to_ndarray(format="gray16le").astype(np.float32)  # (H0, W0)
                break
        return d_u16 / 65535.0 * (hi - lo) + lo  # inverse depth

    def __getitem__(self, index: int):
        item = self.data[index]
        idxs = list(range(self.num_frames))  # first N contiguous frames
        timestamps = item["timestamps"]
        camera_data = item["camera_data"]

        # First image gives us the source resolution and the crop/scale geometry.
        first_path = os.path.join(item["rgb_folder"], f"{timestamps[0]}.jpg")
        with Image.open(first_path) as im0:
            W0, H0 = im0.size
        scale, Hr, Wr, x0, y0 = compute_resize_crop(H0, W0, self.height, self.width)

        rgb = torch.empty(self.num_frames, 3, self.height, self.width, dtype=torch.uint8)
        intrinsics = torch.empty(self.num_frames, 3, 3, dtype=torch.float32)
        w2c = torch.empty(self.num_frames, 4, 4, dtype=torch.float32)

        for out_i, i in enumerate(idxs):
            path = os.path.join(item["rgb_folder"], f"{timestamps[i]}.jpg")
            img = Image.open(path).convert("RGB")
            if scale != 1.0:
                img = img.resize((Wr, Hr), resample=Image.BILINEAR)
            img = img.crop((x0, y0, x0 + self.width, y0 + self.height))
            rgb[out_i] = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1)

            K = self._to_pixel_intrinsic(
                torch.tensor(camera_data[i]["intrinsic"], dtype=torch.float32), W0, H0)
            K[0, 0] *= scale
            K[1, 1] *= scale
            K[0, 2] = K[0, 2] * scale - x0
            K[1, 2] = K[1, 2] * scale - y0
            intrinsics[out_i] = K
            w2c[out_i] = torch.tensor(camera_data[i]["w2c"], dtype=torch.float32)

        # Frame-0 metric depth (N=1 buffer). Depth video is at source resolution -> same geometry.
        with open(item["depth_range_file"]) as f:
            depth_ranges = json.load(f)
        lo, hi = depth_ranges[0]["min_inverse_depth"], depth_ranges[0]["max_inverse_depth"]
        inv_depth0 = self._decode_depth_frame0_inverse(item["depth_video_file"], lo, hi)  # (H0, W0)
        depth0 = torch.from_numpy(1.0 / (inv_depth0 + 1e-6)).unsqueeze(0).unsqueeze(0)  # (1,1,H0,W0)
        if scale != 1.0:
            depth0 = F.interpolate(depth0, size=(Hr, Wr), mode="bilinear", align_corners=False)
        depth0 = depth0[..., y0:y0 + self.height, x0:x0 + self.width]  # (1,1,H,W)

        return {
            "sample_id": item["sample_id"],
            "rgb": rgb,                       # (T,3,H,W) uint8, [0,255]
            "depth0": depth0.squeeze(0),      # (1,H,W) float32 metric z
            "intrinsic": intrinsics,          # (T,3,3) pixel
            "w2c": w2c,                       # (T,4,4)
            "prompt": item["prompt"],
        }
