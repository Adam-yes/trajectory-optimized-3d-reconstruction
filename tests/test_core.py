"""Analytic/synthetic unit tests. None of these are robotic reconstruction experiments."""
import sys

import numpy as np
import pytest
from PIL import Image

from eyeinhand.analysis import (
    bootstrap_mean_interval,
    decile_indices,
    paired_differences,
    pareto_mask,
)
from eyeinhand.artifacts import (
    atomic_json,
    completed_artifacts_match,
    environment_manifest,
    fingerprint,
    read_json,
    run_lease,
    sha256_file,
)
from eyeinhand.cli import main
from eyeinhand.errors import ArtifactConflictError, StageError, ValidationError
from eyeinhand.execution import ExecutionRecord, admission
from eyeinhand.geometry import (
    apply_transform,
    compose_transforms,
    flange_pose,
    look_at,
    points_array,
    polyline_length,
    validate_transform,
)
from eyeinhand.images import Frame, export_prediction_points, prepare_frames, read_frames
from eyeinhand.mesh import load_mesh, sample_surface, validate_mesh
from eyeinhand.metrics import MetricConfig, c2m_statistics, cloud_metrics
from eyeinhand.pipeline import process_bounded, run_command
from eyeinhand.registration import box_prealignment, command_backend, multipass_registration
from eyeinhand.trajectory import (
    DEFAULT_GRID,
    EllipsoidTrajectory,
    grid_size,
    iter_grid,
    snapshot_indices,
)
from eyeinhand.workflow import evaluate_files


@pytest.fixture
def triangle():
    return np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]])


@pytest.mark.parametrize("bad", [[], [[1, 2]], [[1, 2, np.nan]], [[1, 2, np.inf]]])
def test_points_reject_bad(bad):
    with pytest.raises(ValidationError):
        points_array(bad)


def test_transform_order_and_inverse():
    a = np.eye(4); a[:3, 3] = [1, 2, 3]
    b = np.eye(4); b[:3, :3] *= 2
    assert np.allclose(apply_transform([[0, 0, 0]], compose_transforms([a, b])), [[2, 4, 6]])
    assert np.allclose(flange_pose(a, np.eye(4)), a)


@pytest.mark.parametrize("kind", ["reflection", "shear", "anisotropic", "zero", "nan", "row", "tiny"])
def test_transform_reject_invalid(kind):
    t = np.eye(4)
    if kind == "reflection": t[0, 0] = -1
    if kind == "shear": t[0, 1] = .1
    if kind == "anisotropic": t[0, 0] = 2
    if kind == "zero": t[:3, :3] = 0
    if kind == "nan": t[0, 0] = np.nan
    if kind == "row": t[3, 2] = 1
    if kind == "tiny": t[:3, :3] *= 1e-12
    with pytest.raises(ValidationError): validate_transform(t)


def test_scale_limits():
    t = np.eye(4); t[:3, :3] *= 20
    with pytest.raises(ValidationError): validate_transform(t, max_scale=10)
    with pytest.raises(ValidationError): validate_transform(t, rigid=True)


@pytest.mark.parametrize("position", [[0, 0, 1], [0, 0, -1], [1, 2, 3], [1e-9, 0, 1]])
def test_look_at_orthonormal(position):
    t = look_at(position, [0, 0, 0])
    assert np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3))
    assert np.linalg.det(t[:3, :3]) == pytest.approx(1)
    assert np.allclose(t[:3, 2], -np.array(position) / np.linalg.norm(position))


def test_look_at_coincident_and_polyline():
    with pytest.raises(ValidationError): look_at([0, 0, 0], [0, 0, 0])
    assert polyline_length([[0, 0, 0], [3, 4, 0], [3, 4, 2]]) == 7
    assert polyline_length([[0, 0, 0]]) == 0


def test_trajectory_polar_convention_and_grid():
    t = EllipsoidTrajectory(.2, .5, -140, 120, 0, 60)
    p = t.positions()
    assert p.shape == (80, 3)
    assert p[0, 2] == pytest.approx(.54)
    assert np.allclose(p[0, :2], [.35, -.36])
    assert grid_size(DEFAULT_GRID) == 10000
    rows = list(iter_grid(DEFAULT_GRID))
    assert len(rows) == 10000
    assert len(set(tuple(params.values()) for _, params in rows)) == 10000
    assert len(t.poses()) == 80
    features = t.physical_features()
    assert features["max_camera_height_m"] == pytest.approx(.54)


@pytest.mark.parametrize("kwargs", [dict(radius_xy=0), dict(radius_z=-1), dict(elevation_min=-1),
    dict(elevation_max=91), dict(azimuth_end=-160), dict(num_slices=1), dict(points_per_slice=True)])
def test_bad_trajectory(kwargs):
    d = dict(radius_xy=.2, radius_z=.5, azimuth_start=-160, azimuth_end=120,
             elevation_min=5, elevation_max=80)
    d.update(kwargs)
    with pytest.raises(ValidationError): EllipsoidTrajectory(**d)


def test_snapshot_budget():
    idx = snapshot_indices(80, 20)
    assert len(idx) == len(set(idx)) == 20
    assert idx[0] == 0 and idx[-1] == 79
    assert len(snapshot_indices(4, 20)) == 4


def test_perfect_metrics():
    pts = [[0, 0, 0], [1, 0, 0]]
    m = cloud_metrics(pts, pts)
    assert m["f1_score"] == 1
    assert m["hd95_m"] == m["chamfer_unsquared_sum_m"] == 0
    assert m["aucc_tmax_normalized"] == pytest.approx(.99)
    assert m["aucc_interval_normalized"] == pytest.approx(1)


def test_direction_and_boundary_metrics():
    m = cloud_metrics([[0, 0, 0]], [[0, 0, 0], [1, 0, 0]])
    assert m["completeness_ratio"] == 1
    assert m["f1_precision"] == .5
    assert m["f1_score"] == pytest.approx(2/3)
    assert m["cd_forward_gt_to_prediction_m"] == 0
    assert m["cd_backward_prediction_to_gt_m"] == .5
    assert m["hd95_m"] == pytest.approx(.95)


def test_threshold_inclusive_and_zero_f1():
    assert cloud_metrics([[0, 0, 0]], [[.02, 0, 0]])["f1_score"] == 1
    assert cloud_metrics([[0, 0, 0]], [[.1, 0, 0]])["f1_score"] == 0


@pytest.mark.parametrize("kwargs", [dict(threshold_m=0), dict(c2m_cutoff_m=np.nan),
    dict(aucc_min_m=.2), dict(aucc_steps=1), dict(aucc_steps=True)])
def test_bad_metrics_config(kwargs):
    with pytest.raises(ValidationError): MetricConfig(**kwargs)


def test_c2m_discard_not_cap_and_outlier_separate():
    m = c2m_statistics([.01, .05, .15, .3])
    assert m["c2m_conditional_mean_m"] == pytest.approx(.03)
    assert m["c2m_raw_mean_m"] == pytest.approx(.1275)
    assert m["c2m_retained_fraction"] == .5
    assert m["outlier_fraction"] == .25
    assert c2m_statistics([.4])["c2m_conditional_mean_m"] is None
    assert not c2m_statistics([.4])["c2m_conditional_valid"]


def test_pareto_dominated_row_and_ties():
    keep = pareto_mask([4.296, 4.434, 4.296, 3.016], [.899, .878, .899, .851])
    assert keep.tolist() == [True, False, True, True]
    assert pareto_mask([], []).size == 0


def test_deciles_and_pairing():
    indices = decile_indices(np.arange(20)/20)
    assert indices["best"].tolist() == [18, 19]
    assert indices["worst"].tolist() == [0, 1]
    assert indices["middle"].tolist() == [9, 10]
    result = paired_differences({"a": .1, "b": .5}, {"b": .6, "c": .7})
    assert result["n_pairs"] == 1
    assert result["mean_difference"] == pytest.approx(.1)
    assert result["only_a"] == ["a"]
    with pytest.raises(ValidationError): paired_differences({"a": 1}, {"b": 1})


def test_bootstrap_repeatable():
    assert bootstrap_mean_interval([1., 2, 3], seed=5, repetitions=100) == bootstrap_mean_interval([1., 2, 3], seed=5, repetitions=100)


def test_atomic_json_and_strict_read(tmp_path):
    p = tmp_path / "data.json"
    atomic_json(p, {"a": 1})
    assert read_json(p) == {"a": 1}
    with pytest.raises(ValidationError): atomic_json(p, {"a": np.nan})
    assert read_json(p) == {"a": 1}
    for text in ('{"a":1,"a":2}', '{"a":NaN}', 'bad json'):
        p.write_text(text)
        with pytest.raises(ValidationError): read_json(p)


def test_fingerprint_lease_and_cache(tmp_path):
    p = tmp_path / "input"; p.write_text("a")
    d1 = fingerprint({"x": 1}, {"image": p}, software={"version": 1})
    p.write_text("b")
    assert d1 != fingerprint({"x": 1}, {"image": p}, software={"version": 1})
    assert not completed_artifacts_match(tmp_path, d1)
    with run_lease(tmp_path):
        with pytest.raises(ArtifactConflictError):
            with run_lease(tmp_path): pass
    assert not (tmp_path / ".run.lock").exists()
    atomic_json(tmp_path / "manifest.json", {"state": "complete", "fingerprint": d1,
                                          "artifacts": {"input": sha256_file(p)}})
    assert completed_artifacts_match(tmp_path, d1)
    p.write_text("changed")
    assert not completed_artifacts_match(tmp_path, d1)


def test_lease_removed_on_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with run_lease(tmp_path): raise RuntimeError()
    assert not (tmp_path / ".run.lock").exists()


def _frame_manifest(tmp_path, duplicate=False):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (128, 72), (10, 20, 30)).save(a)
    Image.new("RGB", (128, 72), (10 if duplicate else 20, 20, 30)).save(b)
    p = tmp_path / "frames.json"
    atomic_json(p, {"schema": "eyeinhand.frames/1", "frames": [
        {"path": "a.png", "role": "anchor"}, {"path": "b.png", "role": "trajectory"}]})
    return p


def test_image_budget_duplicates_order(tmp_path):
    p = _frame_manifest(tmp_path, True)
    with pytest.raises(ValidationError): read_frames(p, max_total_frames=2)
    frames = read_frames(p, max_total_frames=2, allow_duplicates=True)
    assert len(frames) == 2
    with pytest.raises(ValidationError): read_frames(p, max_total_frames=1, allow_duplicates=True)
    doc = read_json(p); doc["frames"].reverse(); atomic_json(p, doc)
    with pytest.raises(ValidationError): read_frames(p, max_total_frames=2, allow_duplicates=True)


def test_preprocess_mask_and_sizes(tmp_path):
    p = _frame_manifest(tmp_path)
    frames = read_frames(p, max_total_frames=2)
    mask = tmp_path / "mask.png"
    a = np.ones((72, 128), dtype=np.uint8)*255; a[:, :64] = 0
    Image.fromarray(a).save(mask)
    frames[1] = Frame(frames[1].path, "trajectory", mask)
    images, valid, records = prepare_frames(frames)
    assert images.shape == (2, 3, 294, 518)
    assert not valid[1, :, 0].any()
    assert np.all(images[1, :, :, 0] == 0)
    assert records[1]["mask_sha256"] is not None
    assert valid[0].all()


def test_bad_mask(tmp_path):
    p = _frame_manifest(tmp_path)
    frames = read_frames(p, max_total_frames=2)
    mask = tmp_path / "mask.png"; Image.new("L", (128, 72), 127).save(mask)
    with pytest.raises(ValidationError): prepare_frames([Frame(frames[0].path, "anchor", mask)])
    Image.new("L", (20, 20), 255).save(mask)
    with pytest.raises(ValidationError): prepare_frames([Frame(frames[0].path, "anchor", mask)])


def test_export_no_dummy_no_masked_points():
    pts = np.ones((1, 2, 2, 3)); conf = np.ones((1, 2, 2)); rgb = np.ones((1, 3, 2, 2))
    mask = np.array([[[1, 0], [1, 0]]], dtype=bool)
    output, colors, metadata = export_prediction_points(pts, conf, rgb, mask)
    assert len(output) == len(colors) == 2
    assert metadata["retained_points"] == 2
    with pytest.raises(ValidationError): export_prediction_points(pts, conf*0, rgb, mask)


@pytest.mark.parametrize("change,reason", [({"planning_fraction": .75}, "planning_fraction_not_strictly_above_threshold"),
    ({"action_succeeded": False}, "action_failed"), ({"fresh_images": False}, "stale_or_unverified_images"),
    ({"images_saved": 0}, "no_images"), ({"poses_saved": 2}, "image_pose_count_mismatch"),
    ({"process_returncode": 1}, "process_error")])
def test_admission_conjunction(change, reason):
    d = dict(planning_fraction=.8, action_succeeded=True, images_saved=20, poses_saved=20,
             fresh_images=True, process_returncode=0)
    assert admission(ExecutionRecord(**d))["accepted"]
    d.update(change)
    result = admission(ExecutionRecord(**d))
    assert not result["accepted"] and reason in result["reasons"]
    assert not result["collision_safety_certified"]


def test_bounded_pipeline_failure_propagation():
    assert process_bounded(range(20), lambda i: i*i, queue_size=1) == [i*i for i in range(20)]
    def bad():
        yield 1
        raise RuntimeError("producer crash")
    with pytest.raises(StageError): process_bounded(bad(), lambda x: x)
    def bad_consumer(x): raise RuntimeError("consumer crash")
    with pytest.raises(RuntimeError): process_bounded(range(10000), bad_consumer, queue_size=1)


def test_process_exit_and_timeout(tmp_path):
    log = tmp_path / "process.log"
    result = run_command([sys.executable, "-c", "print('ok')"], cwd=tmp_path, log_path=log, timeout_s=5)
    assert result["returncode"] == 0 and "ok" in log.read_text()
    with pytest.raises(StageError): run_command([sys.executable, "-c", "raise SystemExit(3)"], cwd=tmp_path, log_path=log, timeout_s=5)
    with pytest.raises(StageError): run_command([sys.executable, "-c", "import time; time.sleep(5)"], cwd=tmp_path, log_path=log, timeout_s=.1)
    with pytest.raises(ValidationError): run_command("echo bad", cwd=tmp_path, log_path=log, timeout_s=5)


def test_mesh_sampling_reproducible(triangle, tmp_path):
    v, f = triangle
    p1 = sample_surface(v, f, count=100, seed=3)
    assert np.array_equal(p1, sample_surface(v, f, count=100, seed=3))
    assert (p1 >= 0).all() and np.all(p1[:, 0]+p1[:, 1] <= 1)
    path = tmp_path / "mesh.npz"; np.savez(path, vertices=v, faces=f)
    scaled, _ = load_mesh(path, meters_per_unit=.001)
    assert scaled[1, 0] == .001
    with pytest.raises(ValidationError): validate_mesh(v, [[0, 0, 0]])


def test_mesh_scene_transform_applied(tmp_path):
    import trimesh
    scene = trimesh.Scene()
    transform = np.eye(4); transform[0, 3] = 10
    scene.add_geometry(trimesh.creation.box(), transform=transform)
    path = tmp_path / "scene.glb"; scene.export(path)
    vertices, _ = load_mesh(path, meters_per_unit=1)
    assert vertices[:, 0].mean() == pytest.approx(10)


def test_registration_composition_no_double_initial():
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=float)
    initial = np.eye(4); initial[0, 3] = 2
    def backend(src, tgt, index):
        t = np.eye(4); t[1, 3] = 1
        return t
    total, stages = multipass_registration(p, p, backend, initial=initial)
    assert np.allclose(total[:3, 3], [2, 2, 0])
    assert len(stages) == 3


def test_pca_box_alignment():
    rng = np.random.default_rng(1)
    p = rng.uniform(-1, 1, (500, 3)) * [1, 2, 4]
    target = p * 2 + [4, 5, 6]
    transform, metadata = box_prealignment(p, target)
    assert np.allclose(apply_transform(p, transform), target, atol=1e-6)
    assert metadata["candidate_count"] == 24


def test_command_backend_contract(tmp_path):
    identity = {"commit": "test-fixture", "checkpoint_sha256": "0"*64}
    script = tmp_path / "fake_backend.py"
    script.write_text("import sys,json\nimport numpy as np\nfrom pathlib import Path\n"
        "out=Path(sys.argv[sys.argv.index('--output')+1])\n"
        f"out.write_text(json.dumps({{'convention':'T_target_source','group':'SE3','transform':np.eye(4).tolist(),'software_identity':{identity!r}}}))\n")
    fn = command_backend([sys.executable, str(script)], directory=tmp_path / "passes", timeout_s=10, software_identity=identity)
    assert np.allclose(fn(np.ones((3, 3)), np.ones((3, 3)), 0), np.eye(4))


def test_workflow_cache_and_provenance(tmp_path, triangle):
    v, f = triangle
    mesh = tmp_path / "mesh.npz"; np.savez(mesh, vertices=v, faces=f)
    pred = tmp_path / "pred.npz"; np.savez(pred, points=sample_surface(v, f, count=100, seed=0))
    transform = tmp_path / "transform.npy"; np.save(transform, np.eye(4))
    kwargs = dict(meters_per_mesh_unit=1, transform_path=transform, gt_samples=100, exact_c2m=False)
    result = evaluate_files(pred, mesh, tmp_path / "run", **kwargs)
    assert result["f1_score"] == 1
    assert result["c2m_status"] == "not_computed"
    assert evaluate_files(pred, mesh, tmp_path / "run", **kwargs) == result
    with pytest.raises(ArtifactConflictError):
        evaluate_files(pred, mesh, tmp_path / "run", **(kwargs | {"seed": 1}))


def test_cli_demo_and_failure(tmp_path, capsys):
    output = tmp_path / "demo.json"
    assert main(["demo", "--output", str(output)]) == 0
    assert read_json(output)["measurement_run"] is False
    assert main(["doctor"]) == 0
    assert "python" in environment_manifest()
    assert main(["metrics", "--gt", "missing.npy", "--prediction", "missing.npy", "--output", str(output)]) == 2


def test_cli_grid_and_pareto(tmp_path, capsys):
    out = tmp_path / 'grid.json'
    assert main(['grid', '--output', str(out)]) == 0
    data = read_json(out)
    assert data['num_configurations'] == 10000 and not data['feasibility_tested']
    csv_path = tmp_path / 'candidates.csv'
    csv_path.write_text('path_length_m,f1_score\n4.296,.899\n4.434,.878\n')
    assert main(['pareto', '--input', str(csv_path), '--output', str(tmp_path/'front.json')]) == 0
    assert read_json(tmp_path/'front.json')['n_nondominated'] == 1


def test_workflow_failure_is_recorded(tmp_path, triangle):
    v, f = triangle
    mesh = tmp_path/'mesh.npz'; np.savez(mesh, vertices=v, faces=f)
    pred = tmp_path/'bad.npz'; np.savez(pred, points=np.empty((0, 3)))
    transform = tmp_path/'t.npy'; np.save(transform, np.eye(4))
    with pytest.raises(ValidationError):
        evaluate_files(pred, mesh, tmp_path/'run', meters_per_mesh_unit=1,
                       transform_path=transform, exact_c2m=False)
    assert read_json(tmp_path/'run/manifest.json')['state'] == 'failed'
    assert not (tmp_path/'run/.run.lock').exists()


def test_file_cache_rejects_traversal(tmp_path):
    atomic_json(tmp_path/'manifest.json', {'state':'complete','fingerprint':'x',
                                         'artifacts':{'../outside': 'fake'}})
    assert not completed_artifacts_match(tmp_path, 'x')


@pytest.mark.parametrize('patch,width', [(0,518),(-1,518),(14,0),(14,517)])
def test_invalid_preprocessing_dimensions(patch, width):
    with pytest.raises(ValidationError):
        prepare_frames([], patch_size=patch, target_width=width)


@pytest.mark.integration
def test_exact_c2m_optional_backend(triangle):
    pytest.importorskip('open3d', reason='Exact triangle-distance backend needs Open3D')
    from eyeinhand.mesh import exact_surface_distances
    v, f = triangle
    distances = exact_surface_distances(np.array([[.2,.2,.5],[.2,.2,0]]), v, f)
    assert np.allclose(distances, [.5,0], atol=1e-6)
