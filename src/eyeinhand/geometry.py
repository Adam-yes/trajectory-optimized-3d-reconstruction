"""Geometry conventions: row-vector points, column-vector homogeneous transforms.

``T_target_source`` maps source coordinates into target coordinates. A sequence
``[T1, T2]`` is applied as ``T2 @ T1``. Distances use meters after registration.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import ValidationError

FloatArray = NDArray[np.float64]


def points_array(values: ArrayLike, *, name: str = "points", allow_empty: bool = False) -> FloatArray:
    """Validate rather than silently discard invalid geometry."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValidationError(f"{name} must have shape (N, 3); got {arr.shape}")
    if not allow_empty and arr.shape[0] == 0:
        raise ValidationError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValidationError(f"{name} contains non-finite coordinates")
    return arr


def validate_transform(
    value: ArrayLike, *, min_scale: float = 1e-8, max_scale: float | None = None,
    rigid: bool = False, atol: float = 1e-5,
) -> FloatArray:
    """Accept only finite, orientation-preserving Sim(3), or SE(3) if requested."""
    t = np.asarray(value, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValidationError("Transform must be a finite 4x4 matrix")
    if not np.allclose(t[3], [0, 0, 0, 1], atol=atol, rtol=0):
        raise ValidationError("Transform has an invalid homogeneous last row")
    if min_scale <= 0 or not np.isfinite(min_scale):
        raise ValidationError("min_scale must be finite and positive")
    a = t[:3, :3]
    singular = np.linalg.svd(a, compute_uv=False)
    scale = float(singular.mean())
    if scale < min_scale or (max_scale is not None and scale > max_scale):
        raise ValidationError(f"Scale {scale:g} is outside the permitted range")
    if np.linalg.det(a) <= 0 or not np.allclose(singular, scale, rtol=atol, atol=atol*scale):
        raise ValidationError("Reflections, shear and anisotropic scaling are not Sim(3)")
    if rigid and not np.isclose(scale, 1.0, atol=atol, rtol=atol):
        raise ValidationError("A rigid transform must have unit scale")
    return t.copy()


def apply_transform(points: ArrayLike, transform: ArrayLike) -> FloatArray:
    p = points_array(points)
    t = validate_transform(transform)
    return p @ t[:3, :3].T + t[:3, 3]


def compose_transforms(transforms: Iterable[ArrayLike]) -> FloatArray:
    """Compose in application order, exactly once per stage."""
    result = np.eye(4, dtype=np.float64)
    for t in transforms:
        result = validate_transform(t) @ result
    return validate_transform(result)


def polyline_length(points: ArrayLike) -> float:
    """Length between recorded poses; NOT full controller path or elapsed time."""
    p = points_array(points)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def look_at(position: ArrayLike, target: ArrayLike) -> FloatArray:
    """World-from-camera pose with +Z forward and a right-handed orthonormal basis.

    Uses the cross(world_up, forward) convention. This is not an
    assertion about a physical camera's optical frame; use measured extrinsics.
    """
    p = np.asarray(position, dtype=float)
    q = np.asarray(target, dtype=float)
    if p.shape != (3,) or q.shape != (3,) or not np.isfinite([p, q]).all():
        raise ValidationError("position and target must be finite 3-vectors")
    forward = q - p
    norm = np.linalg.norm(forward)
    if norm < 1e-12:
        raise ValidationError("A look-at position cannot equal its target")
    forward /= norm
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(up @ forward)) > 0.999:
        up = np.eye(3)[int(np.argmin(np.abs(forward)))]
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    camera_up = np.cross(forward, right)
    t = np.eye(4)
    t[:3, :3] = np.column_stack((right, camera_up, forward))
    t[:3, 3] = p
    return validate_transform(t, rigid=True)


def flange_pose(world_from_camera: ArrayLike, flange_from_camera: ArrayLike) -> FloatArray:
    """T_world_flange = T_world_camera @ inverse(T_flange_camera)."""
    a = validate_transform(world_from_camera, rigid=True)
    b = validate_transform(flange_from_camera, rigid=True)
    return validate_transform(a @ np.linalg.inv(b), rigid=True)
