"""Run admission is explicit and distinct from planning feasibility or physical safety."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ValidationError


@dataclass(frozen=True)
class ExecutionRecord:
    planning_fraction: float
    action_succeeded: bool
    images_saved: int
    poses_saved: int
    fresh_images: bool
    process_returncode: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.planning_fraction) or not 0 <= self.planning_fraction <= 1:
            raise ValidationError("Planning fraction must be in [0,1]")
        for value in (self.images_saved, self.poses_saved):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError("Capture counts must be nonnegative integers")
        if not isinstance(self.action_succeeded, bool) or not isinstance(self.fresh_images, bool):
            raise ValidationError("Action and image freshness states must be booleans")


def admission(record: ExecutionRecord, *, threshold: float = 0.75) -> dict:
    if not np.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValidationError("Admission threshold must be in [0,1)")
    reasons = []
    if record.process_returncode != 0:
        reasons.append("process_error")
    if not record.action_succeeded:
        reasons.append("action_failed")
    if record.planning_fraction <= threshold:
        reasons.append("planning_fraction_not_strictly_above_threshold")
    if record.images_saved < 1:
        reasons.append("no_images")
    if record.poses_saved != record.images_saved:
        reasons.append("image_pose_count_mismatch")
    if not record.fresh_images:
        reasons.append("stale_or_unverified_images")
    return {"accepted": not reasons, "reasons": reasons, "threshold": threshold,
            "quantity": "planning_fraction", "collision_safety_certified": False}
