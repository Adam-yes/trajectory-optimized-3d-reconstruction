"""Explicit image ordering, independent mask provenance and total-view accounting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .artifacts import atomic_json, read_json, sha256_file
from .errors import ValidationError


@dataclass(frozen=True)
class Frame:
    path: Path
    role: str
    mask: Path | None = None

    def __post_init__(self) -> None:
        if self.role not in {"anchor", "trajectory"}:
            raise ValidationError("Frame role must be 'anchor' or 'trajectory'")
        if not self.path.is_file() or (self.mask is not None and not self.mask.is_file()):
            raise ValidationError(f"Missing frame/mask: {self.path}")


def read_frames(path: str | Path, *, max_total_frames: int, allow_duplicates: bool = False) -> list[Frame]:
    manifest = Path(path).resolve()
    data = read_json(manifest)
    if set(data) != {"schema", "frames"} or data["schema"] != "eyeinhand.frames/1":
        raise ValidationError("Expected an explicit eyeinhand.frames/1 manifest")
    frames = []
    seen_trajectory = False
    seen_hashes = set()
    for row in data["frames"]:
        if set(row) - {"path", "role", "mask"}:
            raise ValidationError("Unknown frame field")
        frame = Frame((manifest.parent / row["path"]).resolve(), row["role"],
                      (manifest.parent / row["mask"]).resolve() if row.get("mask") else None)
        if frame.role == "anchor" and seen_trajectory:
            raise ValidationError("Anchors must precede trajectory frames; order is part of the input")
        seen_trajectory |= frame.role == "trajectory"
        digest = sha256_file(frame.path)
        if digest in seen_hashes and not allow_duplicates:
            raise ValidationError(f"Duplicate image bytes: {frame.path}; explicit override required")
        seen_hashes.add(digest)
        frames.append(frame)
    if isinstance(max_total_frames, bool) or not isinstance(max_total_frames, int) or max_total_frames < 1:
        raise ValidationError("max_total_frames must be a positive integer")
    if not frames or not seen_trajectory or len(frames) > max_total_frames:
        raise ValidationError("Need trajectory frames and a total budget that INCLUDES all anchors")
    return frames


def prepare_frames(frames: list[Frame], *, target_width: int = 518, patch_size: int = 14) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """VGGT-style center-crop geometry; not a claim of bitwise equivalence with upstream.

    RGB is bilinear-resized; masks are nearest-neighbor resized. Black pixels alone
    do not mask transformer attention. The returned validity mask also filters the
    exported 3D points. Mixing aspect ratios is rejected rather than silently padded.
    """
    if (not isinstance(patch_size, int) or isinstance(patch_size, bool) or patch_size <= 0
            or not isinstance(target_width, int) or isinstance(target_width, bool)
            or target_width < patch_size or target_width % patch_size):
        raise ValidationError("target_width must be a positive multiple of patch_size")
    if not frames:
        raise ValidationError("No frames")
    images, masks, records = [], [], []
    for frame in frames:
        with Image.open(frame.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            w, h = image.size
            scaled_h = max(patch_size, round(h / w * target_width / patch_size) * patch_size)
            image = image.resize((target_width, scaled_h), Image.Resampling.BILINEAR)
        if frame.mask is None:
            valid = Image.new("L", (w, h), 255)
        else:
            with Image.open(frame.mask) as m:
                valid = ImageOps.exif_transpose(m).convert("L")
                if valid.size != (w, h):
                    raise ValidationError(f"Mask size does not match image: {frame.path}")
                # Prevent an ambiguous antialiased/soft mask from changing the policy.
                unique = np.unique(np.asarray(valid))
                if not set(unique.tolist()).issubset({0, 1, 255}):
                    raise ValidationError("Masks must be binary; positive pixels are retained")
                valid = valid.point(lambda value: 255 if value else 0)
        valid = valid.resize((target_width, scaled_h), Image.Resampling.NEAREST)
        if scaled_h > target_width:
            top = (scaled_h - target_width) // 2
            box = (0, top, target_width, top + target_width)
            image, valid = image.crop(box), valid.crop(box)
        mask = np.asarray(valid, dtype=np.uint8) > 0
        if not mask.any():
            raise ValidationError(f"Mask removes the entire image: {frame.path}")
        rgb = np.asarray(image, dtype=np.float32) / 255
        rgb *= mask[..., None]
        images.append(rgb.transpose(2, 0, 1))
        masks.append(mask)
        records.append({"path": str(frame.path), "role": frame.role,
                        "sha256": sha256_file(frame.path), "original_size": [w, h],
                        "processed_size": list(image.size),
                        "mask_sha256": sha256_file(frame.mask) if frame.mask else None})
    if len({a.shape for a in images}) != 1:
        raise ValidationError("Processed sizes differ; use a documented common camera/image geometry")
    return np.stack(images), np.stack(masks), records


def export_prediction_points(points: np.ndarray, confidence: np.ndarray, images: np.ndarray,
                             valid: np.ndarray, *, percentile: float = 50.0,
                             min_confidence: float = 1e-5) -> tuple[np.ndarray, np.ndarray, dict]:
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValidationError("Confidence percentile must be in [0,100]")
    if (points.shape != (*valid.shape, 3) or confidence.shape != valid.shape
            or images.shape != (valid.shape[0], 3, *valid.shape[1:])):
        raise ValidationError("Point, image, confidence and mask shapes are inconsistent")
    candidate = valid & np.isfinite(points).all(axis=-1) & np.isfinite(confidence)
    if not candidate.any():
        raise ValidationError("No finite unmasked predictions; no dummy geometry will be emitted")
    threshold = max(float(np.percentile(confidence[candidate], percentile)), min_confidence)
    keep = candidate & (confidence >= threshold) & (confidence > min_confidence)
    if not keep.any():
        raise ValidationError("Confidence filtering removed every point")
    colors = images.transpose(0, 2, 3, 1)
    return points[keep], colors[keep], {"confidence_percentile": percentile,
        "confidence_threshold": threshold, "retained_points": int(keep.sum()),
        "eligible_points": int(candidate.sum()), "total_pixels": int(valid.size)}


def create_study_inputs(trajectory_images: list[Path], *, anchor: Path, gripper_mask: Path,
                        output: Path, anchor_mask: Path | None, max_total_frames: int = 21) -> dict:
    """Build S/I/G/B inputs: one context image, explicit gripper masks, no silent fallback.

    The anchor mask is specified independently: a top view may also contain the gripper.
    Images retain the caller's order. The total budget includes the anchor. Nothing
    is dropped to fit the budget; a mismatch fails before any inference is started.
    """
    import os

    if not trajectory_images:
        raise ValidationError("An ordered trajectory image list is required")
    anchor, gripper_mask, output = Path(anchor).resolve(), Path(gripper_mask).resolve(), Path(output).resolve()
    for path in [anchor, gripper_mask, *trajectory_images]:
        if not Path(path).is_file():
            raise ValidationError(f"Missing study input: {path}")
    if output.exists():
        raise ValidationError("Study output already exists; use a new directory")
    if isinstance(max_total_frames, bool) or not isinstance(max_total_frames, int) or max_total_frames < 2:
        raise ValidationError("Total frame budget must be an integer >= 2")
    if len(trajectory_images) + 1 > max_total_frames:
        raise ValidationError("The budget includes one anchor; no images have been discarded")
    # Validate both modalities before writing any manifests.
    base = [Frame(Path(p).resolve(), "trajectory", gripper_mask) for p in trajectory_images]
    extended = [Frame(anchor, "anchor", Path(anchor_mask).resolve() if anchor_mask else None), *base]
    hashes = [sha256_file(f.path) for f in extended]
    if len(hashes) != len(set(hashes)):
        raise ValidationError("Duplicate image bytes in the study inputs")
    prepare_frames(extended)
    output.mkdir(parents=True)
    variants = {}
    for name, context, initialize in [("Simple", False, False), ("InitialAlign", False, True),
                                     ("GlobalAnchor", True, False), ("Both", True, True)]:
        frames = extended if context else base
        entries = [{"path": os.path.relpath(f.path, output), "role": f.role,
                    **({"mask": os.path.relpath(f.mask, output)} if f.mask else {})} for f in frames]
        filename = f"{name}.frames.json"
        atomic_json(output / filename, {"schema": "eyeinhand.frames/1", "frames": entries})
        read_frames(output / filename, max_total_frames=max_total_frames)
        variants[name] = {"manifest": filename, "initial_alignment": initialize,
                          "bufferx_passes": 2, "teaserpp": False,
                          "trajectory_frames": len(base), "anchor_frames": int(context)}
    result = {"schema": "eyeinhand.study/1", "max_total_frames": max_total_frames,
              "mask_policy": "explicit trajectory and independent anchor masks",
              "anchor_masked": anchor_mask is not None,
              "view_budget_policy": "same trajectory frames; context adds one image",
              "prior_results_recomputed": False, "variants": variants}
    atomic_json(output / "study.json", result)
    return result
