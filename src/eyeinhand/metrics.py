"""Explicit, uncensored bidirectional metrics plus separately labelled C2M summaries.

Empty or non-finite point sets are failures, never perfect reconstructions.
The tmax-normalized AUCC divides by the maximum threshold, not the interval width.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from .errors import ValidationError
from .geometry import FloatArray, points_array

METRIC_SCHEMA = "eyeinhand.metrics/2.0"


def positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValidationError(f"{name} must be finite and positive")
    return float(value)


def distances_array(values: ArrayLike, name: str = "distances") -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all() or (arr < 0).any():
        raise ValidationError(f"{name} must be a nonempty finite, nonnegative 1D array")
    return arr


@dataclass(frozen=True)
class MetricConfig:
    threshold_m: float = 0.02
    c2m_cutoff_m: float = 0.10
    outlier_threshold_m: float = 0.20
    aucc_min_m: float = 0.001
    aucc_max_m: float = 0.10
    aucc_steps: int = 100

    def __post_init__(self) -> None:
        for name in ("threshold_m", "c2m_cutoff_m", "outlier_threshold_m", "aucc_min_m", "aucc_max_m"):
            positive(getattr(self, name), name)
        if self.aucc_min_m >= self.aucc_max_m:
            raise ValidationError("AUCC minimum must be smaller than maximum")
        if isinstance(self.aucc_steps, bool) or not isinstance(self.aucc_steps, int) or self.aucc_steps < 2:
            raise ValidationError("AUCC requires at least two integer threshold steps")


def directional_distances(source: ArrayLike, target: ArrayLike) -> FloatArray:
    source = points_array(source, name="source")
    target = points_array(target, name="target")
    return np.asarray(cKDTree(target).query(source, k=1, workers=1)[0])


def cloud_metrics(gt: ArrayLike, predicted: ArrayLike, config: MetricConfig | None = None) -> dict:
    cfg = config or MetricConfig()
    forward = directional_distances(gt, predicted)
    backward = directional_distances(predicted, gt)
    return metrics_from_distances(forward, backward, cfg)


def metrics_from_distances(forward: ArrayLike, backward: ArrayLike,
                           config: MetricConfig | None = None) -> dict:
    """Summarize GT-to-prediction and prediction-to-GT distances without censoring."""
    cfg = config or MetricConfig()
    forward = distances_array(forward, "GT-to-prediction distances")
    backward = distances_array(backward, "prediction-to-GT distances")
    recall = float(np.mean(forward <= cfg.threshold_m))
    precision = float(np.mean(backward <= cfg.threshold_m))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    thresholds = np.linspace(cfg.aucc_min_m, cfg.aucc_max_m, cfg.aucc_steps)
    coverage = np.searchsorted(np.sort(forward), thresholds, side="right") / len(forward)
    # Trapezoid integration written explicitly for NumPy 1.x and 2.x compatibility.
    area = float(np.sum(np.diff(thresholds) * (coverage[:-1] + coverage[1:]) / 2))
    return {
        "schema": METRIC_SCHEMA, "units": "meters", "f1_threshold_m": cfg.threshold_m,
        "f1_score": f1, "f1_precision": precision, "f1_recall": recall,
        "completeness_ratio": recall, "num_gt_points": len(forward),
        "num_predicted_points": len(backward),
        "cd_forward_gt_to_prediction_m": float(np.mean(forward)),
        "cd_backward_prediction_to_gt_m": float(np.mean(backward)),
        "chamfer_unsquared_sum_m": float(np.mean(forward) + np.mean(backward)),
        "hd95_m": float(max(np.percentile(forward, 95), np.percentile(backward, 95))),
        "aucc_tmax_normalized": area / cfg.aucc_max_m,
        "aucc_interval_normalized": area / (cfg.aucc_max_m - cfg.aucc_min_m),
        "aucc_range_m": [cfg.aucc_min_m, cfg.aucc_max_m], "aucc_steps": cfg.aucc_steps,
    }


def c2m_statistics(raw_distances: ArrayLike, config: MetricConfig | None = None) -> dict:
    """Never label a sampled-cloud distance as exact C2M. Supply true surface distances.

    Both discarded-outlier and uncensored statistics are returned. Null means no
    retained values, and the validity flag prevents null from becoming zero.
    """
    cfg = config or MetricConfig()
    raw = distances_array(raw_distances)
    retained = raw[raw <= cfg.c2m_cutoff_m]
    return {
        "c2m_raw_mean_m": float(raw.mean()),
        "c2m_raw_rmse_m": float(np.sqrt(np.mean(raw**2))),
        "c2m_raw_p95_m": float(np.percentile(raw, 95)),
        "c2m_conditional_mean_m": float(retained.mean()) if retained.size else None,
        "c2m_conditional_valid": bool(retained.size),
        "c2m_cutoff_m": cfg.c2m_cutoff_m, "c2m_truncation_mode": "discard",
        "c2m_retained_fraction": float(retained.size / raw.size),
        "c2m_num_source_points": int(raw.size), "c2m_num_retained_points": int(retained.size),
        "outlier_fraction": float(np.mean(raw > cfg.outlier_threshold_m)),
        "outlier_threshold_m": cfg.outlier_threshold_m,
    }
