# Running the pipeline

How to take an acquisition from a trajectory proposal to a metric evaluation, and how
to choose an operating point from the runs you end up with. Every new run must carry a
new identifier and must not overwrite an earlier result.

## 1. Acquisition and frame preparation

Use `eyeinhand.trajectory.EllipsoidTrajectory` and `iter_grid` for the nominal 10-by-8 viewing pattern and stable grid identifiers. The `elevation_*` field names denote the ellipsoid polar parameter from +Z, not an elevation above the horizontal plane. Camera poses are look-at poses; apply the actual calibrated camera-to-flange transform for the intended setup.

Apply `integrations/ros2_scene.py` to the simulated planning scene. The configured target-object box is a geometric proxy: verify it encloses the actual objects in the loaded scene. Planning validity, successful action completion, fresh images, and valid pose/image pairs are separate checks; use the capture adapter and `ExecutionRecord` admission policy. A scene update alone is not a motion validation.

Prepare the image manifests with `eyeinhand prepare-study`. Input order is explicit. The four settings share the same trajectory frames and gripper mask; context-enabled settings add one selected context frame. Independently specify the anchor mask or explicitly request an unmasked anchor. Duplicate bytes and over-budget sequences are rejected. The usual cap of 20 trajectory images plus one context frame requires a total budget of 21; an equal-total-view ablation needs a separately designed sampling policy.

## 2. Reconstruction

Install the intended upstream VGGT checkout and supply its local checkpoint and SHA-256. Example argument names:

```bash
python -m eyeinhand reconstruct --manifest runs/study_001/Both.frames.json \
  --checkpoint /path/to/model.pt --checkpoint-sha256 SHA256 \
  --upstream-commit COMMIT --max-total-frames 21 \
  --output runs/study_001/vggt
```

`SHA256`, `COMMIT`, and `/path/to/...` are required local values, not bundled identities. The adapter has no automatic checkpoint download and uses explicit image masks. It filters point predictions by validity and confidence. Its image preparation and point export must be treated as part of the experiment configuration.

## 3. Registration

Prepare a metric reference sample from the intended mesh; do not substitute a differently named scene without verifying geometry, units, crop, and transforms. `eyeinhand.mesh.load_mesh` and `sample_surface` provide the mesh and seeded surface-sampling operations. Point archives use the `points` NPZ key.

`eyeinhand register` takes source and target point archives and a JSON backend configuration containing `argv`, `timeout_s`, `commit`, and `checkpoint_sha256`. The argument vector invokes `integrations/bufferx_worker.py` with its trusted local model factory, checkpoint, and configuration. The worker protocol is documented in `integrations/README.md`.

The default `obb` initializer applies an OBB-centroid, sorted-extent scale, axis alignment, and a fixed local half-turn. It requires Open3D. An axis and sign-search initializer is available through the explicit `pca-search` option. Neither method establishes metric scale from RGB alone: both use the reference geometry. Two incremental rigid transforms are composed once after the initializer.

The command saves `transform.npy`, per-pass data, and `registration.json`. Record the backend commit and checkpoint hash with every run; a changed backend is a different experiment.

## 4. Evaluation

```bash
python -m eyeinhand evaluate --prediction runs/study_001/vggt/points.npz \
  --mesh /path/to/reference_mesh.npz --meters-per-mesh-unit 1 \
  --transform runs/study_001/registration/transform.npy \
  --gt-samples 1000000 --seed 0 --output runs/study_001/evaluation
```

The exact triangle-distance backend requires Open3D. An explicit `--skip-c2m` computes the other metrics without substituting nearest-point distances for C2M. Outputs include metrics, the aligned point cloud, and `distances.npz` with uncensored directional distances and, when evaluated, point-to-mesh distances. C2M cutoff and outlier threshold are independent. Output hashes participate in cache checks.

## 5. Choosing an operating point

Collect the runs you want to compare into a CSV with a `run_id` column plus
`path_length_m`, `f1_score` and `hd95_m`, and add `radius_xy_m`, `radius_z_m`,
`polar_min_deg` and `polar_max_deg` if you also want the viewing geometry.

```python
from eyeinhand.operating_points import (
    GEOMETRY_COLUMNS, RUN_COLUMNS, load_runs, nondominated_costs,
    select_operating_point, sightline_geometry, supported_f1_length_points,
)

rows = load_runs("runs/summary.csv", RUN_COLUMNS + GEOMETRY_COLUMNS)
best = select_operating_point(rows, min_f1=.85, max_hd95_m=.12)
front = nondominated_costs([[r["path_length_m"], -r["f1_score"], r["hd95_m"]] for r in rows])
supported = supported_f1_length_points(rows)
geometry = sightline_geometry(rows, focal_height_m=.04)
```

Pick `min_f1` and `max_hd95_m` from the tolerance of the downstream task rather than by
tuning. Tasks that act on object boundaries are sensitive to the tail of the surface
error, which is what HD95 tracks; a coarse collision map is not. Comparing `front` with
`supported` shows which runs a single weighted objective could never return, so a run
that is nondominated but unsupported is one that scalarizing would silently discard.

These are exact calculations over the rows you supply. They carry no significance test,
no confidence interval and no causal ranking of the trajectory parameters.
