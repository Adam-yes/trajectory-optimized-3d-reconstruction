"""Explicit reference-assisted registration with validated stage transforms."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from .artifacts import atomic_json, read_json, sha256_file
from .errors import ValidationError
from .geometry import apply_transform, compose_transforms, points_array, validate_transform
from .pipeline import run_command


def _pca_box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(points - origin, rowvar=False))
    axes = vectors[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    local = (points - origin) @ axes
    low, high = local.min(axis=0), local.max(axis=0)
    extents = high - low
    if (extents < 1e-10).any():
        raise ValidationError("Degenerate PCA box; use an explicitly supplied initial transform")
    center = origin + axes @ ((low + high) / 2)
    return center, axes, extents


def box_prealignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    """PCA-box initializer with axis and sign enumeration; not the OBB convention.

    Uses ground-truth geometry and selects the minimum symmetric NN mean on
    deterministic subsamples. This is an evaluation prior, not metric inference
    from RGB. Its results must not be mixed with results produced by the OBB initializer.
    """
    src, tgt = points_array(source), points_array(target)
    cs, rs, es = _pca_box(src)
    ct, rt, et = _pca_box(tgt)
    small_s = src[np.linspace(0, len(src)-1, min(len(src), 2048), dtype=int)]
    small_t = tgt[np.linspace(0, len(tgt)-1, min(len(tgt), 2048), dtype=int)]
    target_tree = cKDTree(small_t)
    best = None
    candidates = 0
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            destination_axes = rt[:, perm] * np.asarray(signs)
            rotation = destination_axes @ rs.T
            if np.linalg.det(rotation) < 0:
                continue
            scale = float(np.mean(et[list(perm)] / es))
            transform = np.eye(4)
            transform[:3, :3] = scale * rotation
            transform[:3, 3] = ct - scale * rotation @ cs
            aligned = apply_transform(small_s, transform)
            score = float(target_tree.query(aligned)[0].mean()
                          + cKDTree(aligned).query(small_t)[0].mean())
            candidates += 1
            if best is None or score < best[0]:
                best = score, transform
    assert best is not None
    return best[1], {"method": "pca-box-axis-search/v2", "candidate_count": candidates,
                     "selection_score_symmetric_nn_m": best[0], "uses_reference_geometry": True}


def multipass_registration(source: np.ndarray, target: np.ndarray,
                           backend: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
                           *, passes: int = 2, initial: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < 1:
        raise ValidationError("passes must be a positive integer")
    original, target = points_array(source), points_array(target)
    total = validate_transform(np.eye(4) if initial is None else initial)
    stages = [{"stage": "initial", "transform": total.tolist()}]
    for index in range(passes):
        current = apply_transform(original, total)
        incremental = validate_transform(backend(current, target, index), rigid=True)
        total = compose_transforms([total, incremental])
        stages.append({"stage": f"registration_pass_{index+1}",
                       "transform_current_to_target": incremental.tolist()})
    return total, stages


def command_backend(argv: list[str], *, directory: str | Path, timeout_s: float,
                    software_identity: dict) -> Callable:
    """Isolate a pinned registration implementation behind an NPZ/JSON protocol.

    Executable receives --input INPUT.npz --output OUTPUT.json. Its output must
    state T_target_source and SE3 explicitly. No implicit transform inversion.
    """
    if not software_identity.get("commit") or not software_identity.get("checkpoint_sha256"):
        raise ValidationError("Backend identity requires an exact commit and checkpoint SHA-256")
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def run(source: np.ndarray, target: np.ndarray, index: int) -> np.ndarray:
        input_path, output_path = root / f"pass_{index}.npz", root / f"pass_{index}.json"
        if output_path.exists() or input_path.exists():
            raise ValidationError("Backend pass artifacts already exist; use an isolated run directory")
        np.savez_compressed(input_path, source=source, target=target)
        run_command([*argv, "--input", str(input_path), "--output", str(output_path)],
                    cwd=root, log_path=root / f"pass_{index}.log", timeout_s=timeout_s)
        result = read_json(output_path)
        if result.get("convention") != "T_target_source" or result.get("group") != "SE3":
            raise ValidationError("Backend must return convention T_target_source and group SE3")
        if result.get("software_identity") != software_identity:
            raise ValidationError("Backend's actual software/checkpoint identity does not match configuration")
        atomic_json(root / f"pass_{index}_provenance.json", {
            "input_sha256": sha256_file(input_path), "output_sha256": sha256_file(output_path),
            "software_identity": software_identity})
        return validate_transform(result["transform"], rigid=True)
    return run


def obb_box_transform(source_box: tuple, target_box: tuple,
                         *, local_half_turn: bool = True) -> tuple[np.ndarray, dict]:
    """Oriented-bounding-box prealignment with an explicit box-axis convention.

    Each box is (center[3], right-handed axes[3,3], positive extents[3]).
    The fixed local half-turn disambiguates the chosen OBB convention; it is
    NOT an optimization over alternative axes. Metric scale uses the reference.
    """
    def validated(box):
        if len(box) != 3:
            raise ValidationError("An OBB requires center, axes and extents")
        center, axes, extents = (np.asarray(x, dtype=float) for x in box)
        if center.shape != (3,) or axes.shape != (3, 3) or extents.shape != (3,):
            raise ValidationError("Invalid OBB array shapes")
        if not all(np.isfinite(x).all() for x in (center, axes, extents)) or (extents <= 1e-10).any():
            raise ValidationError("An OBB must be finite and nondegenerate")
        t = np.eye(4); t[:3, :3] = axes
        validate_transform(t, rigid=True)
        return center, axes, extents
    cs, rs, es = validated(source_box)
    ct, rt, et = validated(target_box)
    scale = float(np.mean(np.sort(et) / np.sort(es)))
    rotation = rt @ rs.T
    if local_half_turn:
        rotation = (rt @ np.diag([-1., -1., 1.]) @ rt.T) @ rotation
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = ct - scale * rotation @ cs
    return validate_transform(transform), {
        "method": "obb/v1", "scale": scale,
        "local_half_turn": local_half_turn, "uses_reference_geometry": True}


def obb_prealignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    """Compute Open3D OBBs and apply the fixed half-turn initializer."""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ValidationError("The OBB initializer requires the geometry extra (Open3D)") from exc
    def box(points):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points_array(points))
        try:
            obb = cloud.get_oriented_bounding_box(robust=False)
        except RuntimeError as exc:
            raise ValidationError("Cannot determine a nondegenerate oriented bounding box") from exc
        return np.asarray(obb.center), np.asarray(obb.R), np.asarray(obb.extent)
    return obb_box_transform(box(source), box(target))
