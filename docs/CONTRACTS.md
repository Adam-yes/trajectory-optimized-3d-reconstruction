# Data and software contracts

## Geometry and units

`T_target_source` is a 4x4 column-vector homogeneous transform mapping source coordinates into the target frame. Points are stored as row vectors and transformed as `P @ T[:3,:3].T + T[:3,3]`. Applying T1, then T2 gives `T2 @ T1`. The initializer is applied exactly once. Rigid registration passes must return SE(3); reflections, shear, anisotropic scaling, non-finite values and near-zero scale are rejected.

The `elevation_*` fields are **polar angles from positive Z**, not elevation above the horizontal plane. The default grid specifies ten azimuth slices and eight polar samples (80 geometric waypoints), then caps recorded images at 20 controller samples. Angle metadata is in degrees; trigonometric calculations use radians.

`polyline_length` measures the provided samples only, taken from recorded `wrist_3_link` poses. Do not rename it dense camera travel, energy, or cycle time.

## Image manifest

```json
{
  "schema": "eyeinhand.frames/1",
  "frames": [
    {"path": "anchor.png", "role": "anchor"},
    {"path": "images/000.png", "role": "trajectory", "mask": "gripper_keep_mask.png"}
  ]
}
```

Paths are relative to the manifest. Ordering is explicit and unchanged. Anchors must precede trajectory frames. Exact duplicate image bytes require an explicit override; their multiplicity is not silently changed. `max_total_frames` counts **all** frames, including anchors. Masks use positive pixels to retain scene content and zero pixels to exclude it. They must have the original image size; interpolation is nearest-neighbor. A mask belongs to an individual frame, not automatically to every camera.

The new preprocessing is documented as a new implementation: 518-pixel width, height rounded to a multiple of 14, centered crop for tall inputs. Raw 1280x720 captures become 518x294. Pixel zeroing is not an attention-mask guarantee. The validity mask is also used when exporting points. Empty predictions fail rather than inserting a dummy point. Export is direct from point maps, avoiding camera visualizers and GLB scene-graph ambiguity.

## Metric schema `eyeinhand.metrics/2.0`

F1/precision/recall use an inclusive 0.02 m distance threshold by default. Distances are unsquared. Forward means ground truth to prediction; backward means prediction to ground truth. Chamfer is the **sum**, not the mean, of directional means. HD95 is the maximum of the two directional 95th percentiles, not a percentile over pooled distances.

C2M is a distance to triangle surfaces, not a sampled reference cloud. Conditional mean discards points beyond a configurable cutoff, default 0.10 m. Always retain the uncensored mean/RMSE/p95 and retained fraction. Outliers have an independent threshold, default 0.20 m. If no points remain after truncation, conditional mean is JSON `null`, accompanied by `c2m_conditional_valid: false`, never zero or nonstandard NaN.

AUCC reports two names: `aucc_tmax_normalized` divides the integral on [0.001,0.10] by 0.10; `aucc_interval_normalized` divides by 0.099. A perfectly covered cloud therefore has tmax-normalized AUCC 0.99, not 1. They must not be mixed.

## Registration worker protocol

A backend runs in a separately pinned environment:

```text
python integrations/bufferx_worker.py --factory my_pinned_build:make_model \
  --checkpoint /absolute/path/model.pt --checkpoint-sha256 ACTUAL_SHA256 \
  --commit ACTUAL_COMMIT --seed 0 --input pass.npz --output pass.json
```

Input NPZ keys are `source` and `target`, each finite N x 3. Output JSON must contain `convention: T_target_source`, `group: SE3`, `transform`, and `software_identity` with the exact `commit` and `checkpoint_sha256`. The factory constructs the intended model and configuration **without downloading weights**. A factory cannot be supplied truthfully for an unspecified external package revision; retrieve that revision before integration testing. The worker then strictly loads and verifies the supplied local checkpoint. The implementation is not silently substituted with a newer BUFFER-X variant.

The command backend runs two incremental passes by default. Checkpoint/factory errors are stage failures, never an identity registration success. Model input hashes, logs and each incremental transform are preserved.

## Run manifests

A run uses a directory lease. Atomic JSON records `running`, `complete`, or `failed`. Resume is allowed only if the config/software/input fingerprint matches and every recorded output hash still matches. Incomplete or mismatched outputs require a new directory; failure artifacts are never silently deleted. Leases surviving a crash must be inspected before manual removal.
