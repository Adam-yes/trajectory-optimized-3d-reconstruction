"""Finite-set selection over recorded cost-quality operating points.

A scan family produces many executable trajectories, and each one is described by a
motion cost and several disagreeing quality measures. The functions here compare such
a finite set exactly: Pareto dominance in any number of objectives, selection of the
cheapest run that satisfies explicit quality thresholds, the subset a weighted sum can
ever return, and the viewing geometry implied by each recorded configuration.

Every result is an exact calculation over the rows you pass in. Nothing here fits a
model, resamples a distribution, or infers a confidence interval, so none of it
supports a significance claim about trajectories that were never executed.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .analysis import pareto_mask
from .errors import ValidationError

__all__ = ["RUN_COLUMNS", "GEOMETRY_COLUMNS", "load_runs", "nondominated_costs",
           "select_operating_point", "supported_f1_length_points", "sightline_geometry"]

#: Columns every run table must provide, beyond the ``run_id`` key.
RUN_COLUMNS = ("path_length_m", "f1_score", "hd95_m")
#: Extra columns required to derive viewing geometry from a run.
GEOMETRY_COLUMNS = ("radius_xy_m", "radius_z_m", "polar_min_deg", "polar_max_deg")


def load_runs(path: Path, required: tuple[str, ...] = RUN_COLUMNS) -> list[dict]:
    """Read a run table and return its rows sorted by acquisition-path length.

    Values are converted to floats except for ``run_id``. Rows are rejected rather than
    repaired: duplicate identifiers, missing columns, non-finite numbers and values
    outside the metric domain all raise instead of being silently dropped.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValidationError(f"No rows in {path}")
    missing = {"run_id", *required} - set(raw[0])
    if missing:
        raise ValidationError(f"Run table is missing columns: {sorted(missing)}")
    rows = []
    for row in raw:
        converted = {}
        for key, value in row.items():
            if key == "run_id":
                converted[key] = value
                continue
            try:
                converted[key] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"Non-numeric value in column {key!r}") from exc
        rows.append(converted)
    if len({r["run_id"] for r in rows}) != len(rows):
        raise ValidationError("Run identifiers must be unique")
    for r in rows:
        if not all(np.isfinite(v) for k, v in r.items() if k != "run_id"):
            raise ValidationError("Non-finite run data")
        if not 0 <= r["f1_score"] <= 1 or r["path_length_m"] <= 0 or r["hd95_m"] < 0:
            raise ValidationError("Run data outside the metric domain")
    return sorted(rows, key=lambda r: r["path_length_m"])


def nondominated_costs(costs: np.ndarray) -> np.ndarray:
    """Mask of rows that no other row improves on in every minimized objective."""
    values = np.asarray(costs, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValidationError("Costs must be a finite, nonempty matrix")
    return np.array([not np.any(np.all(values <= row, axis=1) & np.any(values < row, axis=1))
                     for row in values], dtype=bool)


def select_operating_point(rows: list[dict], min_f1: float, max_hd95_m: float) -> dict | None:
    """Shortest run meeting both quality thresholds, or ``None`` when none does.

    The thresholds express a quality preference taken from the downstream task. They
    are not clearance or safety guarantees.
    """
    if not np.isfinite(min_f1) or not 0 <= min_f1 <= 1 \
            or not np.isfinite(max_hd95_m) or max_hd95_m < 0:
        raise ValidationError("Invalid F1/HD95 preference")
    feasible = [r for r in rows if r["f1_score"] >= min_f1 and r["hd95_m"] <= max_hd95_m]
    if not feasible:
        return None
    return min(feasible, key=lambda r: (r["path_length_m"], -r["f1_score"], r["run_id"]))


def supported_f1_length_points(rows: list[dict]) -> list[str]:
    """Upper concave envelope: points attainable by maximizing F1 minus lambda*length.

    Scalarizing cost and quality into one weighted objective can only ever return a run
    on this envelope, so a nondominated run that is absent from the result is one a
    weighted sum would never select at any nonnegative weight. Collinear points are
    retained; lambda is nonnegative for these increasing fronts.
    """
    keep = pareto_mask([r["path_length_m"] for r in rows], [r["f1_score"] for r in rows])
    hull: list[dict] = []
    for row, retain in zip(rows, keep):
        if not retain:
            continue
        while len(hull) >= 2:
            a, b = hull[-2:]
            previous = (b["f1_score"]-a["f1_score"])/(b["path_length_m"]-a["path_length_m"])
            new = (row["f1_score"]-b["f1_score"])/(row["path_length_m"]-b["path_length_m"])
            if new <= previous:
                break
            hull.pop()
        hull.append(row)
    return [r["run_id"] for r in hull]


def sightline_geometry(rows: list[dict], focal_height_m: float = 0.0) -> list[dict]:
    """Camera height range and physical sightline inclination for each run.

    The polar parameter of an ellipsoid is not the angle the camera actually looks
    along. With horizontal radius r_xy and vertical radius r_z, a polar parameter theta
    corresponds to an inclination from vertical of ``arctan((r_xy/r_z) tan(theta))``,
    and the two coincide only on a sphere. ``focal_height_m`` offsets the reported
    heights to the frame the focal target is expressed in.
    """
    missing = set(GEOMETRY_COLUMNS) - set(rows[0] if rows else {})
    if missing:
        raise ValidationError(f"Run table is missing geometry columns: {sorted(missing)}")
    result = []
    for r in rows:
        low, high = np.deg2rad([r["polar_min_deg"], r["polar_max_deg"]])
        if not 0 <= low <= high < np.pi/2:
            raise ValidationError("Polar limits must satisfy 0 <= min <= max < 90 degrees")
        ratio = r["radius_xy_m"]/r["radius_z_m"]
        result.append({"run_id": r["run_id"],
                       "height_min_m": focal_height_m + r["radius_z_m"]*np.cos(high),
                       "height_max_m": focal_height_m + r["radius_z_m"]*np.cos(low),
                       "sightline_polar_min_deg": float(np.rad2deg(np.arctan(ratio*np.tan(low)))),
                       "sightline_polar_max_deg": float(np.rad2deg(np.arctan(ratio*np.tan(high))))})
    return result
