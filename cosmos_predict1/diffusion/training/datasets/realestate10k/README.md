# RealEstate10K → GEN3C data prep (Step 1)

Turns Voyager's raw RealEstate10K into a GEN3C training-ready **latent** cache.
No existing GEN3C files are modified.

## What gets cached (per clip, `<output_dir>/<set_name>/<sample_id>.pt`)

| key | shape | meaning |
|---|---|---|
| `video_latent` | `(16,16,88,160)` fp16 | target video latent = `tokenizer.encode(gt)*sigma_data`. Also the video2world condition source (training masks its first `num_condition_t` latent frames — no separate condition tensor). |
| `condition_pose_latent` | `(64,16,88,160)` fp16 | GEN3C warped-frame condition = `encode_warped_frames` of the frame-0 3D-cache render. Layout `[v0(16), m0(16), v1(16=0), m1(16=0)]`: buffer 0 = real, buffer 1 = zero-padded (N=1 strategy). |
| `t5_text_embeddings` | `(L,1024)` fp16 | T5-XXL, trimmed to true length (GEN3C convention; pad to 512 in the training loader). |
| `t5_text_mask` | `(L,)` int64 | all ones (length `L`). |
| `meta` | dict | sample_id, prompt, shapes, sigma_data, frame_buffer_max, filter_points_threshold. |

## Run (on the GPU server)

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python \
  cosmos_predict1/diffusion/training/datasets/realestate10k/build_gen3c_cache.py \
  --dataset-root /path/to/realestate10k \
  --set-name refined_train_10pct \
  --checkpoint-dir checkpoints \
  --output-dir datasets/gen3c_re10k \
  --limit 3            # start small, then drop --limit for the full run
```

Then **verify before the mass run** (decodes latents back to mp4, GT on top / warped-condition below):

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python \
  cosmos_predict1/diffusion/training/datasets/realestate10k/verify_cache.py \
  --cache-dir datasets/gen3c_re10k/refined_train_10pct \
  --out-dir datasets/gen3c_re10k/_verify --num 3
```

## Design choices (locked)

- **N=1 buffers**: warp only frame 0 to all 121 target cameras; buffer 2 zero-padded. Matches GEN3C
  single-image inference's *first* 121-frame chunk and the Step-2 single-video sanity goal.
- **Latent cache** (not pixels): warped pixel buffers are ~1.3 GB/clip; latents ~37 MB/clip.
- **720→704 = resize-to-cover + center-crop**: for exactly 1280×720 this is a pure 8px top/bottom crop
  (`cy-=8`), no resampling of RGB or depth.
- Warp matches inference defaults: `frame_buffer_max=2`, `filter_points_threshold=0.05`,
  `noise_aug_strength=0.0`, `is_depth=True`. `align_depth` is never triggered (update_cache only).

## Verify on the first real run (can't be checked offline)

1. **Depth scale / units.** Metric z = `1/inverse_depth`. `Cache3D_Base.__init__` clamps depth to
   `[0, 100]`. If RealEstate's reconstruction scale puts typical z outside that range the parallax
   breaks — the warped-condition video in `verify_cache` is the tell (frame 0 always looks perfect
   since target cam 0 == source cam; watch frames 1..120 for correct parallax + disocclusion holes).
2. **Intrinsics units.** The loader auto-converts normalized→pixel intrinsics via a `fx<1` heuristic;
   confirm `intrinsic[0]` has `fx≈` hundreds–thousands, `cx≈640`.
3. **Caption format.** Reuses `train_caption.json`, taking the substring before the first `;`.
4. **T5 model.** Defaults to `google-t5/t5-11b` (T5-XXL, d_model=1024) per `scripts/get_t5_embeddings.py`.

## Not in scope (later steps)

N=2 multi-source buffers for long/autoregressive finetuning; training-loop wiring +
`condition_location`/`num_condition_t` (Step 2); dual-branch (Step 4). The cache serves both the
normal and dual-branch runs unchanged.
