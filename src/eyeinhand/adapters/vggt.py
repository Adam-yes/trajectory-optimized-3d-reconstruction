"""Offline VGGT inference. Requires the separately installed, pinned upstream package.

This adapter is source-audited but NOT GPU-integration-tested in this delivery.
No weights are downloaded and no placeholder points/camera meshes are exported.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from ..artifacts import sha256_file
from ..errors import MissingDependencyError, ValidationError
from ..images import export_prediction_points, prepare_frames, read_frames


def reconstruct(manifest: str | Path, *, checkpoint: str | Path, expected_sha256: str,
                upstream_commit: str, max_total_frames: int, seed: int = 0,
                device: str = "cuda", allow_duplicates: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    if not upstream_commit or len(expected_sha256) != 64:
        raise ValidationError("Supply exact upstream commit and checkpoint SHA-256")
    if sha256_file(checkpoint) != expected_sha256:
        raise ValidationError("VGGT checkpoint digest mismatch")
    frames = read_frames(manifest, max_total_frames=max_total_frames, allow_duplicates=allow_duplicates)
    rgb, valid, records = prepare_frames(frames)
    try:
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError as exc:
        raise MissingDependencyError("Install a pinned upstream VGGT checkout and PyTorch") from exc
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValidationError("CUDA requested but unavailable; no implicit CPU fallback")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # GPU kernel determinism is not guaranteed merely by seed control; record policy.
    model = VGGT()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    tensor = torch.from_numpy(rgb).to(device)
    with torch.inference_mode():
        prediction = model(tensor)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(prediction["pose_enc"], tensor.shape[-2:])
        depth = prediction["depth"][0].detach().float().cpu().numpy()
        confidence = prediction["depth_conf"][0].detach().float().cpu().numpy()
        ext = extrinsics[0].detach().float().cpu().numpy()
        intr = intrinsics[0].detach().float().cpu().numpy()
        points = np.asarray(unproject_depth_map_to_point_map(depth, ext, intr))
    points, colors, filtering = export_prediction_points(points, confidence, rgb, valid)
    return points, colors, {"backend": "VGGT", "upstream_commit_declared": upstream_commit,
        "checkpoint_sha256": expected_sha256, "seed": seed, "device": device,
        "precision": "float32", "deterministic_algorithms_enforced": False,
        "frames": records, "num_total_frames": len(frames),
        "num_anchor_frames": sum(f.role == "anchor" for f in frames),
        "preprocessing": "eyeinhand.explicit-mask/2", "coordinate_units": "arbitrary",
        "filtering": filtering, "validated_against_reference_metrics": False}
