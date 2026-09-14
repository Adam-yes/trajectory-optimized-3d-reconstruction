"""Small, composable CLI; --help is safe without ROS, CUDA or model weights."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .analysis import pareto_mask
from .artifacts import atomic_json, environment_manifest, read_json
from .errors import EyeInHandError
from .geometry import points_array
from .metrics import cloud_metrics
from .trajectory import DEFAULT_GRID, grid_size, iter_grid


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eyeinhand", description="Trajectory design and evaluation for eye-in-hand reconstruction")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Print installed versions; does not certify GPU/ROS integration")
    g = sub.add_parser("grid", help="Write geometric grid metadata, NOT feasible trajectories")
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--config", type=Path)
    m = sub.add_parser("metrics", help="Bidirectional metrics for already aligned metric point arrays")
    m.add_argument("--gt", type=Path, required=True)
    m.add_argument("--prediction", type=Path, required=True)
    m.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("pareto", help="Minimize path_length_m, maximize f1_score in a supplied CSV")
    q.add_argument("--input", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    f = sub.add_parser("validate-frames", help="Verify explicit order, total budget and duplicate policy")
    f.add_argument("--manifest", type=Path, required=True)
    f.add_argument("--max-total-frames", type=int, required=True)
    f.add_argument("--allow-duplicates", action="store_true")
    study = sub.add_parser("prepare-study", help="Create single-anchor, masked S/I/G/B input manifests")
    study.add_argument("--images", type=Path, nargs="+", required=True, help="Trajectory files in input order")
    study.add_argument("--anchor", type=Path, required=True)
    study.add_argument("--mask", type=Path, required=True)
    anchor_policy = study.add_mutually_exclusive_group(required=True)
    anchor_policy.add_argument("--anchor-mask", type=Path, help="Mask for the top-down image, if occluded")
    anchor_policy.add_argument("--unmasked-anchor", action="store_true", help="Explicitly retain every anchor pixel")
    study.add_argument("--max-total-frames", type=int, default=21)
    study.add_argument("--output", type=Path, required=True)
    d = sub.add_parser("demo", help="CPU-only synthetic smoke test; NOT a measurement run")
    d.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("evaluate", help="Evaluate saved NPZ predictions against a specified triangle mesh")
    e.add_argument("--prediction", type=Path, required=True)
    e.add_argument("--mesh", type=Path, required=True)
    e.add_argument("--meters-per-mesh-unit", type=float, required=True)
    e.add_argument("--transform", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--gt-samples", type=int, default=100000)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--skip-c2m", action="store_true")
    r = sub.add_parser("reconstruct", help="Offline VGGT; requires separately installed upstream code/weights")
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--checkpoint-sha256", required=True)
    r.add_argument("--upstream-commit", required=True)
    r.add_argument("--max-total-frames", type=int, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--device", default="cuda")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--allow-duplicates", action="store_true")
    reg = sub.add_parser("register", help="Two-pass pinned registration with an explicit initializer")
    reg.add_argument("--source", type=Path, required=True)
    reg.add_argument("--target", type=Path, required=True, help="Metric reference NPZ/NPY point array")
    reg.add_argument("--initializer", choices=["none", "obb", "pca-search"], default="obb")
    reg.add_argument("--backend-config", type=Path, required=True,
                     help="JSON containing argv, timeout_s, commit, checkpoint_sha256")
    reg.add_argument("--output", type=Path, required=True)
    return p


def _load_points(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return points_array(data["points"])
    return points_array(np.load(path, allow_pickle=False))


def dispatch(args: argparse.Namespace) -> dict:
    if args.command == "prepare-study":
        from .images import create_study_inputs
        return create_study_inputs(args.images, anchor=args.anchor, gripper_mask=args.mask,
                                   output=args.output, anchor_mask=args.anchor_mask, max_total_frames=args.max_total_frames)
    if args.command == "register":
        from .artifacts import run_lease, sha256_file
        from .errors import ValidationError
        from .registration import (
            box_prealignment,
            command_backend,
            multipass_registration,
            obb_prealignment,
        )
        config = read_json(args.backend_config)
        if set(config) != {"argv", "timeout_s", "commit", "checkpoint_sha256"}:
            raise ValidationError("Backend config requires argv, timeout_s, commit, checkpoint_sha256")
        if not isinstance(config["argv"], list) or not config["argv"] or not all(
                isinstance(v, str) and v for v in config["argv"]):
            raise ValidationError("Backend argv must be a nonempty list of strings")
        identity = {k: config[k] for k in ("commit", "checkpoint_sha256")}
        source, target = _load_points(args.source), _load_points(args.target)
        initial, initial_info = np.eye(4), {"method": "none"}
        if args.initializer != "none":
            initialize = obb_prealignment if args.initializer == "obb" else box_prealignment
            initial, initial_info = initialize(source, target)
        with run_lease(args.output) as root:
            if any(p.name != ".run.lock" for p in root.iterdir()):
                raise ValidationError("Registration output is not empty")
            backend = command_backend(config["argv"], directory=root / "passes",
                                      timeout_s=config["timeout_s"], software_identity=identity)
            transform, stages = multipass_registration(source, target, backend, initial=initial)
            np.save(root / "transform.npy", transform)
            result = {"schema": "eyeinhand.registration/1", "initializer": initial_info,
                      "stages": stages, "software_identity": identity,
                      "source_sha256": sha256_file(args.source), "target_sha256": sha256_file(args.target),
                      "transform_sha256": sha256_file(root / "transform.npy")}
            atomic_json(root / "registration.json", result)
        return result
    if args.command == "doctor":
        return environment_manifest()
    if args.command == "grid":
        grid = read_json(args.config) if args.config else DEFAULT_GRID
        rows = [{"run_id": run_id, "parameters": params} for run_id, params in iter_grid(grid)]
        atomic_json(args.output, {"schema": "eyeinhand.grid/2", "num_configurations": len(rows),
                    "feasibility_tested": False, "angle_convention": "polar_from_positive_z_degrees", "rows": rows})
        return {"num_configurations": len(rows), "output": str(args.output)}
    if args.command == "metrics":
        result = cloud_metrics(_load_points(args.gt), _load_points(args.prediction))
        atomic_json(args.output, result)
        return result
    if args.command == "pareto":
        with args.input.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        keep = pareto_mask([float(r["path_length_m"]) for r in rows], [float(r["f1_score"]) for r in rows])
        result = {"scope": "supplied rows only; not an unseen full-dataset frontier", "n_input": len(rows),
                  "n_nondominated": int(keep.sum()), "rows": [r for r, k in zip(rows, keep) if k]}
        atomic_json(args.output, result)
        return result
    if args.command == "validate-frames":
        from .images import read_frames
        frames = read_frames(args.manifest, max_total_frames=args.max_total_frames,
                             allow_duplicates=args.allow_duplicates)
        return {"total_frames": len(frames), "anchor_frames": sum(f.role == "anchor" for f in frames),
                "duplicate_override": args.allow_duplicates}
    if args.command == "demo":
        from .geometry import polyline_length
        from .trajectory import EllipsoidTrajectory
        rng = np.random.default_rng(7)
        gt = rng.uniform(-0.1, 0.1, (1000, 3))
        predicted = gt + np.array([0.003, 0, 0])
        trajectory = EllipsoidTrajectory(0.22, 0.50, -140, 120, 10, 65)
        result = {"synthetic_smoke_test_only": True, "measurement_run": False,
                  "metrics": cloud_metrics(gt, predicted), "grid_size": grid_size(DEFAULT_GRID),
                  "planned_camera_waypoints": len(trajectory.positions()),
                  "planned_camera_polyline_m": polyline_length(trajectory.positions())}
        atomic_json(args.output, result)
        return result
    if args.command == "evaluate":
        from .workflow import evaluate_files
        return evaluate_files(args.prediction, args.mesh, args.output,
              meters_per_mesh_unit=args.meters_per_mesh_unit, transform_path=args.transform,
              gt_samples=args.gt_samples, seed=args.seed, exact_c2m=not args.skip_c2m)
    if args.command == "reconstruct":
        from .adapters.vggt import reconstruct
        from .artifacts import run_lease, sha256_file
        from .errors import ArtifactConflictError
        with run_lease(args.output) as root:
            if (root / "points.npz").exists() or (root / "inference.json").exists():
                raise ArtifactConflictError("Inference output already exists; use a new directory")
            points, colors, metadata = reconstruct(args.manifest, checkpoint=args.checkpoint,
                expected_sha256=args.checkpoint_sha256, upstream_commit=args.upstream_commit,
                max_total_frames=args.max_total_frames, seed=args.seed, device=args.device,
                allow_duplicates=args.allow_duplicates)
            np.savez_compressed(root / "points.npz", points=points, colors=colors)
            metadata["points_sha256"] = sha256_file(root / "points.npz")
            atomic_json(root / "inference.json", metadata)
        return {"points": len(points), "output": str(root)}
    raise ValueError(f"Unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (EyeInHandError, OSError, KeyError, ValueError) as exc:
        print(f"eyeinhand: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
