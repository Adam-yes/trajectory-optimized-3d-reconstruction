"""Reference-assisted evaluation of an explicitly reconstructed point cloud."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .artifacts import (
    atomic_json,
    completed_artifacts_match,
    environment_manifest,
    fingerprint,
    run_lease,
    sha256_file,
)
from .errors import ArtifactConflictError, ValidationError
from .geometry import apply_transform, points_array, validate_transform
from .mesh import exact_surface_distances, load_mesh, sample_surface
from .metrics import MetricConfig, c2m_statistics, directional_distances, metrics_from_distances


def evaluate_files(prediction: str | Path, mesh: str | Path, output: str | Path,
                   *, meters_per_mesh_unit: float, transform_path: str | Path,
                   gt_samples: int = 100000, seed: int = 0,
                   exact_c2m: bool = True, metric_config: MetricConfig | None = None) -> dict:
    """Accept only a supplied reference-assisted alignment, never pretend it was inferred.

    Predictions use NPZ key 'points'. Transform is a 4x4 .npy mapping predictions to
    metric reference coordinates. Existing output can only be reused by exact digest.
    """
    cfg = metric_config or MetricConfig()
    config = {"metrics": asdict(cfg), "meters_per_mesh_unit": meters_per_mesh_unit,
              "gt_samples": gt_samples, "seed": seed, "exact_c2m": exact_c2m}
    software = environment_manifest()
    files = {"prediction": prediction, "reference_mesh": mesh, "transform": transform_path}
    initial_hashes = {key: sha256_file(path) for key, path in files.items()}
    digest = fingerprint(config, files, software=software)
    with run_lease(output) as root:
        if completed_artifacts_match(root, digest):
            from .artifacts import read_json
            return read_json(root / "metrics.json")
        if (root / "manifest.json").exists() or (root / "metrics.json").exists():
            raise ArtifactConflictError("Existing run differs or is incomplete; preserve it and use a new directory")
        atomic_json(root / "manifest.json", {"state": "running", "fingerprint": digest,
                    "config": config, "software": software, "inputs": {
                        key: sha256_file(path) for key, path in files.items()}})
        try:
            with np.load(prediction, allow_pickle=False) as archive:
                points = points_array(archive["points"])
            transform = validate_transform(np.load(transform_path, allow_pickle=False))
            aligned = apply_transform(points, transform)
            vertices, faces = load_mesh(mesh, meters_per_unit=meters_per_mesh_unit)
            gt = sample_surface(vertices, faces, count=gt_samples, seed=seed)
            forward = directional_distances(gt, aligned)
            backward = directional_distances(aligned, gt)
            distances = {"gt_to_prediction_m": forward, "prediction_to_gt_m": backward}
            metrics = metrics_from_distances(forward, backward, cfg)
            if exact_c2m:
                distances["prediction_to_mesh_m"] = exact_surface_distances(aligned, vertices, faces)
                metrics.update(c2m_statistics(distances["prediction_to_mesh_m"], cfg))
                metrics["c2m_backend"] = "open3d.triangle_raycast"
            else:
                metrics["c2m_status"] = "not_computed"
            metrics["evaluation_scope"] = "reference_assisted; no autonomous metric-scale claim"
            metrics["gt_sampling_seed"] = seed
            if any(sha256_file(path) != initial_hashes[key] for key, path in files.items()):
                raise ValidationError("An input changed while evaluation was running")
            atomic_json(root / "metrics.json", metrics)
            np.savez_compressed(root / "aligned_points.npz", points=aligned)
            # Keep all distances, not only the points surviving the C2M cutoff.
            np.savez_compressed(root / "distances.npz", **distances)
            artifacts = {name: sha256_file(root / name) for name in ("metrics.json", "aligned_points.npz", "distances.npz")}
            atomic_json(root / "manifest.json", {"state": "complete", "fingerprint": digest,
                "config": config, "software": software, "artifacts": artifacts,
                "inputs": {key: sha256_file(path) for key, path in files.items()}})
            return metrics
        except BaseException as exc:
            atomic_json(root / "manifest.json", {"state": "failed", "fingerprint": digest,
                "error_type": type(exc).__name__, "message": str(exc),
                "config": config, "software": software})
            raise
