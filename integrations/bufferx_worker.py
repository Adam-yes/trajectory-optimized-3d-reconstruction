"""Offline BUFFER-X bridge for the original package API.

NOT integration-tested here. A trusted local factory must construct
(model, cfg) for the exact installed package WITHOUT downloading/loading weights.
Factory signature: make_model(device: str) -> tuple[torch.nn.Module, config].
The boundary is explicit: supply the factory and the package revision yourself.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import random
from pathlib import Path

import numpy as np

from eyeinhand.artifacts import atomic_json, sha256_file
from eyeinhand.errors import ValidationError
from eyeinhand.geometry import points_array, validate_transform


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--factory', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--state-key', help='Explicit checkpoint dictionary key, if not a bare state_dict')
    p.add_argument('--commit', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValidationError('Refusing to overwrite an existing registration result')
    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise ValidationError('Checkpoint SHA-256 mismatch')
    if ':' not in args.factory:
        raise ValidationError('Factory must have the form trusted.module:function')
    import open3d as o3d
    import torch
    from bufferx.dataset.dataloader import collate_fn_descriptor
    from bufferx.utils.tools import sphericity_based_voxel_analysis

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValidationError('CUDA was requested but is unavailable')
    module, name = args.factory.rsplit(':', 1)
    model, config = getattr(importlib.import_module(module), name)(device=args.device)
    cfg = copy.deepcopy(config)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if args.state_key:
        state = state[args.state_key]
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    with np.load(args.input, allow_pickle=False) as data:
        source = points_array(data['source'])
        target = points_array(data['target'])
    def cloud(points):
        c = o3d.geometry.PointCloud()
        c.points = o3d.utility.Vector3dVector(points)
        return c
    src, tgt = cloud(source), cloud(target)
    downsample, sphericity, _ = sphericity_based_voxel_analysis(src, tgt)
    if not np.isfinite(downsample) or downsample <= 0:
        raise ValidationError('BUFFER-X preprocessing returned an invalid voxel size')
    cfg.data.downsample = downsample
    first_s, first_t = src.voxel_down_sample(downsample), tgt.voxel_down_sample(downsample)
    voxel = float(cfg.data.voxel_size_0)
    rng = np.random.default_rng(args.seed)
    def shuffled(c, cap=None):
        array = points_array(np.asarray(c.points))
        indices = rng.permutation(len(array))
        return array[indices[:cap]] if cap is not None else array[indices]
    sample = {
        'src_fds_pts': shuffled(first_s), 'tgt_fds_pts': shuffled(first_t),
        'src_sds_pts': shuffled(first_s.voxel_down_sample(voxel), cfg.data.max_numPts),
        'tgt_sds_pts': shuffled(first_t.voxel_down_sample(voxel), cfg.data.max_numPts),
        'relt_pose': np.eye(4, dtype=np.float32), 'src_id': 0, 'tgt_id': 1,
        'voxel_size': voxel, 'dataset_name': cfg.data.dataset, 'sphericity': sphericity,
        'is_aligned_to_global_z': cfg.patch.is_aligned_to_global_z,
    }
    data = collate_fn_descriptor([sample], cfg)
    with torch.inference_mode():
        transform, timing, src_corr, tgt_corr = model(data)
    def as_numpy(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    transform = validate_transform(as_numpy(transform), rigid=True)
    correspondences_path = args.output.with_suffix('.correspondences.npz')
    np.savez_compressed(correspondences_path,
                        source=as_numpy(src_corr), target=as_numpy(tgt_corr))
    atomic_json(args.output, {
        'convention': 'T_target_source', 'group': 'SE3', 'transform': transform.tolist(),
        'software_identity': {'commit': args.commit, 'checkpoint_sha256': args.checkpoint_sha256},
        'factory': args.factory, 'seed': args.seed, 'input_sha256': sha256_file(args.input),
        'correspondence_sha256': sha256_file(correspondences_path),
        'keypoint_voxel_size': voxel, 'uses_global_z_prior': bool(cfg.patch.is_aligned_to_global_z),
        'deterministic_gpu_execution_verified': False,
    })


if __name__ == '__main__':
    main()
