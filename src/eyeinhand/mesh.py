"""Mesh loading honors scene-graph transforms; sampling is seeded and surface-based."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import MissingDependencyError, ValidationError
from .geometry import points_array


def validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = points_array(vertices, name="mesh vertices")
    f = np.asarray(faces)
    if (f.ndim != 2 or f.shape[1] != 3 or f.size == 0
            or not np.issubdtype(f.dtype, np.integer) or f.min() < 0 or f.max() >= len(v)):
        raise ValidationError("Faces must be nonempty integer triangle indices into vertices")
    tri = v[f]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValidationError("Mesh has no finite nondegenerate surface")
    return v, f.astype(np.int64)


def load_mesh(path: str | Path, *, meters_per_unit: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValidationError("An explicit positive meters_per_unit is required")
    p = Path(path)
    if p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=False) as data:
            vertices, faces = data["vertices"], data["faces"]
    else:
        try:
            import trimesh
        except ImportError as exc:
            raise MissingDependencyError("Install eyeinhand[geometry] for mesh file loading") from exc
        loaded = trimesh.load(p, force="scene", process=False)
        vertices_list, faces_list, offset = [], [], 0
        for node in loaded.graph.nodes_geometry:
            transform, name = loaded.graph[node]
            geom = loaded.geometry[name]
            if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
                continue
            v = trimesh.transform_points(geom.vertices, transform)
            vertices_list.append(v)
            faces_list.append(np.asarray(geom.faces) + offset)
            offset += len(v)
        if not vertices_list:
            raise ValidationError("Scene contains no triangle mesh")
        vertices, faces = np.concatenate(vertices_list), np.concatenate(faces_list)
    return validate_mesh(np.asarray(vertices) * meters_per_unit, faces)


def sample_surface(vertices: np.ndarray, faces: np.ndarray, *, count: int, seed: int) -> np.ndarray:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValidationError("Surface sample count must be a positive integer")
    v, f = validate_mesh(vertices, faces)
    triangles = v[f]
    weights = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                     triangles[:, 2]-triangles[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    chosen = triangles[rng.choice(len(f), count, p=weights / weights.sum())]
    r = rng.random((count, 2))
    root = np.sqrt(r[:, 0])
    return ((1-root)[:, None] * chosen[:, 0] + (root*(1-r[:, 1]))[:, None] * chosen[:, 1]
            + (root*r[:, 1])[:, None] * chosen[:, 2])


def exact_surface_distances(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Exact triangle-distance query in Open3D; fail rather than fall back to sampled NN."""
    points = points_array(points)
    vertices, faces = validate_mesh(vertices, faces)
    try:
        import open3d as o3d
    except ImportError as exc:
        raise MissingDependencyError("Open3D is required for exact C2M; no approximate fallback") from exc
    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(faces.astype(np.int32)))
    scene.add_triangles(mesh)
    result = scene.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy().astype(float)
    if not np.isfinite(result).all():
        raise ValidationError("C2M backend returned invalid distances")
    return result
