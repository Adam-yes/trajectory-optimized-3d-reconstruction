"""Descriptive summaries; no silent pooling of failed, mismatched or unpaired runs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil

import numpy as np
from numpy.typing import ArrayLike

from .errors import ValidationError


def pareto_mask(cost: ArrayLike, quality: ArrayLike) -> np.ndarray:
    """Minimize cost and maximize quality. Equal objective vectors remain nondominated."""
    x, y = np.asarray(cost, dtype=float), np.asarray(quality, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or not np.isfinite([x, y]).all():
        raise ValidationError("Pareto inputs must be equal-length finite vectors")
    if (x < 0).any():
        raise ValidationError("Path cost cannot be negative")
    keep = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        dominates = (x <= x[i]) & (y >= y[i]) & ((x < x[i]) | (y > y[i]))
        keep[i] = not bool(np.any(dominates))
    return keep


def decile_indices(f1: ArrayLike) -> dict[str, np.ndarray]:
    """Prospective convention: ceil(10% n) rows, stable ranking, centered middle band.

    This convention is NOT claimed to match any particular upstream reducer.
    """
    scores = np.asarray(f1, dtype=float)
    if scores.ndim != 1 or len(scores) < 10 or not np.isfinite(scores).all():
        raise ValidationError("Deciles require at least ten finite run scores")
    if ((scores < 0) | (scores > 1)).any():
        raise ValidationError("F1 must be in [0,1]")
    order = np.argsort(scores, kind="stable")
    n = ceil(len(order) / 10)
    middle = (len(order) - n) // 2
    return {"worst": order[:n], "middle": order[middle:middle+n], "best": order[-n:]}


def paired_differences(a: Mapping[str, float], b: Mapping[str, float]) -> dict:
    """B minus A on explicit common run IDs, with all attrition counts exposed."""
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValidationError("No common run IDs; a paired comparison is impossible")
    da = np.asarray([a[k] for k in shared], dtype=float)
    db = np.asarray([b[k] for k in shared], dtype=float)
    if not np.isfinite([da, db]).all():
        raise ValidationError("Paired inputs must not contain NaN/Inf")
    d = db - da
    return {"run_ids": shared, "n_pairs": len(shared), "only_a": sorted(set(a)-set(b)),
            "only_b": sorted(set(b)-set(a)), "mean_difference": float(d.mean()),
            "median_difference": float(np.median(d)), "differences": d.tolist()}


def bootstrap_mean_interval(values: Sequence[float], *, seed: int, repetitions: int = 2000) -> dict:
    """Run-level descriptive bootstrap; correlated runs are NOT independent scenes."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or v.size < 2 or not np.isfinite(v).all():
        raise ValidationError("Bootstrap needs at least two finite observations")
    if repetitions < 100:
        raise ValidationError("Use at least 100 bootstrap repetitions")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, size=v.size, replace=True).mean() for _ in range(repetitions)])
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(v.mean()), "interval_95_percentile": [float(low), float(high)],
            "seed": seed, "repetitions": repetitions, "resampling_unit": "run"}
