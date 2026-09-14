"""Offline viewing geometry. This module does NOT perform IK or certify safety."""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from math import prod

import numpy as np

from .errors import ValidationError
from .geometry import FloatArray, look_at

DEFAULT_GRID: dict[str, list[float]] = {
    "radius_xy": [0.18, 0.20, 0.22, 0.24, 0.26],
    "radius_z": [0.35, 0.40, 0.45, 0.50, 0.55],
    "azimuth_start": [-160.0, -140.0, -120.0, -100.0],
    "azimuth_end": [60.0, 80.0, 100.0, 120.0],
    "elevation_min": [5.0, 10.0, 15.0, 20.0, 25.0],
    "elevation_max": [60.0, 65.0, 70.0, 75.0, 80.0],
}


@dataclass(frozen=True)
class EllipsoidTrajectory:
    """The ``elevation_*`` fields are polar angles from +Z, in degrees, not elevations."""
    radius_xy: float
    radius_z: float
    azimuth_start: float
    azimuth_end: float
    elevation_min: float
    elevation_max: float
    num_slices: int = 10
    points_per_slice: int = 8
    target: tuple[float, float, float] = (0.35, -0.36, 0.04)

    def __post_init__(self) -> None:
        numeric = [self.radius_xy, self.radius_z, self.azimuth_start, self.azimuth_end,
                   self.elevation_min, self.elevation_max, *self.target]
        if len(self.target) != 3 or not np.isfinite(numeric).all():
            raise ValidationError("Trajectory parameters and 3D target must be finite")
        if self.radius_xy <= 0 or self.radius_z <= 0:
            raise ValidationError("Ellipsoid radii must be positive")
        if not 0 <= self.elevation_min < self.elevation_max <= 90:
            raise ValidationError("Polar angles must satisfy 0 <= min < max <= 90 degrees")
        if not 0 < self.azimuth_end - self.azimuth_start <= 360:
            raise ValidationError("Azimuth interval must span (0, 360] degrees")
        for name in ("num_slices", "points_per_slice"):
            n = getattr(self, name)
            if isinstance(n, bool) or not isinstance(n, int) or n < 2:
                raise ValidationError(f"{name} must be an integer >= 2")

    def positions(self) -> FloatArray:
        azimuth = np.deg2rad(np.linspace(self.azimuth_start, self.azimuth_end, self.num_slices))
        polar = np.deg2rad(np.linspace(self.elevation_min, self.elevation_max, self.points_per_slice))
        points = []
        for i, a in enumerate(azimuth):
            for e in polar if i % 2 == 0 else polar[::-1]:
                points.append([self.radius_xy * np.sin(e) * np.cos(a),
                               self.radius_xy * np.sin(e) * np.sin(a),
                               self.radius_z * np.cos(e)])
        return np.asarray(points) + np.asarray(self.target)

    def poses(self) -> FloatArray:
        return np.stack([look_at(p, self.target) for p in self.positions()])

    def physical_features(self) -> dict[str, float]:
        """Correct polar-angle features; no invented ellipsoid surface-area proxy."""
        low, high = np.deg2rad([self.elevation_min, self.elevation_max])
        return {"azimuth_span_deg": self.azimuth_end - self.azimuth_start,
                "max_camera_height_m": self.target[2] + self.radius_z * np.cos(low),
                "vertical_span_m": self.radius_z * (np.cos(low) - np.cos(high))}


def iter_grid(grid: Mapping[str, Sequence[float]] | None = None) -> Iterator[tuple[str, dict]]:
    grid = DEFAULT_GRID if grid is None else grid
    if set(grid) != set(DEFAULT_GRID):
        raise ValidationError("Grid must contain exactly the six documented geometric parameters")
    keys = sorted(grid)
    values = [list(grid[k]) for k in keys]
    if any(not v for v in values):
        raise ValidationError("Grid axes may not be empty")
    if any(len(set(v)) != len(v) for v in values):
        raise ValidationError("Grid axes contain duplicate values")
    for i, combination in enumerate(product(*values)):
        params = dict(zip(keys, combination))
        EllipsoidTrajectory(**params)
        yield f"grid_run_{i:04d}", params


def grid_size(grid: Mapping[str, Sequence[float]] = DEFAULT_GRID) -> int:
    return prod(len(v) for v in grid.values())


def snapshot_indices(trajectory_points: int, max_images: int) -> list[int]:
    for value in (trajectory_points, max_images):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValidationError("Point and image counts must be positive integers")
    return np.linspace(0, trajectory_points - 1, min(trajectory_points, max_images), dtype=int).tolist()
