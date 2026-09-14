"""CPU tests for the input, registration, evaluation and selection workflow."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eyeinhand import operating_points as f
from eyeinhand.artifacts import read_json
from eyeinhand.cli import main
from eyeinhand.errors import ValidationError
from eyeinhand.images import create_study_inputs, prepare_frames, read_frames
from eyeinhand.metrics import cloud_metrics, directional_distances, metrics_from_distances
from eyeinhand.registration import obb_box_transform
from eyeinhand.workflow import evaluate_files

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def images(tmp_path):
    paths = []
    for i in range(4):
        p = tmp_path / f"frame{i}.png"
        Image.fromarray(np.full((28, 42, 3), 30+i, dtype=np.uint8)).save(p)
        paths.append(p)
    mask = np.ones((28, 42), dtype=np.uint8)*255
    mask[:, :14] = 0
    path = tmp_path / "mask.png"
    Image.fromarray(mask).save(path)
    return paths, path


def test_study_single_anchor_and_trajectory_only_mask(images, tmp_path):
    paths, mask = images
    folder = tmp_path / "study"
    result = create_study_inputs(paths[:3], anchor=paths[3], gripper_mask=mask, output=folder, anchor_mask=None)
    assert result["variants"]["Both"]["anchor_frames"] == 1
    assert result["variants"]["Simple"]["anchor_frames"] == 0
    frames = read_frames(folder / "Both.frames.json", max_total_frames=4)
    assert frames[0].mask is None
    assert all(f.mask == mask for f in frames[1:])
    rgb, valid, _ = prepare_frames(frames, target_width=42)
    assert valid[0].all()
    assert not valid[1:, :, :14].any()
    assert not rgb[1:, :, :, :14].any()
    assert [f.path for f in frames[1:]] == paths[:3]


@pytest.mark.parametrize("failure", ["budget", "duplicate", "missing", "mask_shape", "existing"])
def test_study_rejects_bad_inputs(images, tmp_path, failure):
    paths, mask = images
    folder = tmp_path / "study"
    kwargs = dict(trajectory_images=paths[:3], anchor=paths[3], gripper_mask=mask, output=folder, anchor_mask=None)
    if failure == "budget": kwargs["max_total_frames"] = 3
    if failure == "duplicate": kwargs["anchor"] = paths[0]
    if failure == "missing": kwargs["anchor"] = tmp_path / "missing.png"
    if failure == "mask_shape": Image.new("L", (3, 3), 255).save(mask)
    if failure == "existing": folder.mkdir()
    with pytest.raises(ValidationError): create_study_inputs(**kwargs)
    assert not (folder / "study.json").exists()


def test_obb_scale_centroid_and_halfturn():
    source = ([1, 2, 3], np.eye(3), [2, 3, 4])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    target = ([-1, 0, 4], rotation, [4, 6, 8])
    a, meta = obb_box_transform(source, target, local_half_turn=False)
    assert meta["scale"] == 2
    assert np.allclose(a[:3, :3], 2*rotation)
    assert np.allclose(a[:3, :3] @ np.array(source[0])+a[:3, 3], target[0])
    b, _ = obb_box_transform(source, target)
    assert np.allclose(b[:3, :3], 2*rotation @ np.diag([-1, -1, 1]))
    assert np.linalg.det(b[:3, :3]) == pytest.approx(8)


@pytest.mark.parametrize("failure", ["flat", "left_handed", "nonfinite", "shape"])
def test_obb_rejects_invalid(failure):
    src = ([0, 0, 0], np.eye(3), [1, 2, 3])
    tgt = list(copy.deepcopy(src))
    if failure == "flat": tgt[2] = [1, 2, 0]
    if failure == "left_handed": tgt[1] = np.diag([1, 1, -1])
    if failure == "nonfinite": tgt[0] = [0, 0, np.nan]
    if failure == "shape": tgt[1] = np.eye(2)
    with pytest.raises(ValidationError): obb_box_transform(src, tuple(tgt))


def scene_module():
    spec = importlib.util.spec_from_file_location("ros2_scene", ROOT / "integrations/ros2_scene.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collision_scene_explicit_target_and_three_attached_zones():
    spec = read_json(ROOT / "configs/workcell_scene.json")
    scene_module().validate_scene(spec)
    assert "target_objects" in [b["id"] for b in spec["world_boxes"]]
    assert len(spec["attached_boxes"]) == 3
    assert spec["attachment_link"] == "wrist_3_link"


@pytest.mark.parametrize("failure", ["missing_target", "negative_size", "duplicate", "nonfinite"])
def test_collision_scene_rejects_invalid(failure):
    spec = read_json(ROOT / "configs/workcell_scene.json")
    if failure == "missing_target": spec["world_boxes"] = [b for b in spec["world_boxes"] if b["id"] != "target_objects"]
    if failure == "negative_size": spec["world_boxes"][0]["size_m"][0] = -1
    if failure == "duplicate": spec["world_boxes"].append(copy.deepcopy(spec["world_boxes"][0]))
    if failure == "nonfinite": spec["world_boxes"][0]["center_m"][0] = float("nan")
    with pytest.raises(ValidationError): scene_module().validate_scene(spec)


def test_metrics_from_saved_distances_match():
    gt = np.array([[0, 0, 0], [1, 0, 0]])
    pred = np.array([[.03, 0, 0], [1.1, 0, 0]])
    assert cloud_metrics(gt, pred) == metrics_from_distances(directional_distances(gt, pred), directional_distances(pred, gt))


def test_evaluate_archives_uncensored_distances(tmp_path):
    np.savez(tmp_path / "pred.npz", points=[[0, 0, .01], [0, 0, .30]])
    np.savez(tmp_path / "mesh.npz", vertices=[[0,0,0], [1,0,0], [0,1,0]], faces=[[0,1,2]])
    np.save(tmp_path / "t.npy", np.eye(4))
    evaluate_files(tmp_path / "pred.npz", tmp_path / "mesh.npz", tmp_path / "out",
                   meters_per_mesh_unit=1, transform_path=tmp_path / "t.npy", gt_samples=10, exact_c2m=False)
    with np.load(tmp_path / "out/distances.npz") as data:
        assert len(data["prediction_to_gt_m"]) == 2
        assert data["prediction_to_gt_m"].max() >= .3
    manifest = read_json(tmp_path / "out/manifest.json")
    assert "distances.npz" in manifest["artifacts"]


def test_register_cli_protocol(tmp_path):
    """A fake identity backend exercises file protocol, NOT BUFFER-X performance."""
    points = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float)
    np.save(tmp_path / "p.npy", points)
    worker = tmp_path / "worker.py"
    worker.write_text('import argparse,json\np=argparse.ArgumentParser();p.add_argument("--input");p.add_argument("--output");a=p.parse_args()\njson.dump({"convention":"T_target_source","group":"SE3","transform":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],"software_identity":{"commit":"test","checkpoint_sha256":"test"}},open(a.output,"w"))\n')
    config = dict(argv=[sys.executable, str(worker)], timeout_s=5, commit="test", checkpoint_sha256="test")
    (tmp_path / "backend.json").write_text(json.dumps(config))
    argv = ["register", "--source", str(tmp_path/"p.npy"), "--target", str(tmp_path/"p.npy"),
            "--initializer", "none", "--backend-config", str(tmp_path/"backend.json"), "--output", str(tmp_path/"out")]
    assert main(argv) == 0
    assert np.allclose(np.load(tmp_path/"out/transform.npy"), np.eye(4))
    assert len(read_json(tmp_path/"out/registration.json")["stages"]) == 3
    assert main(argv) == 2


def test_anchor_can_use_its_own_gripper_mask(images, tmp_path):
    paths, mask = images
    folder = tmp_path / "anchored"
    result = create_study_inputs(paths[:3], anchor=paths[3], gripper_mask=mask,
                                 output=folder, anchor_mask=mask)
    assert result["anchor_masked"]
    frames = read_frames(folder/"Both.frames.json", max_total_frames=4)
    assert frames[0].mask == mask


SYNTHETIC_RUNS = [
    # id,      length, F1,    HD95,  r_xy, r_z,  theta_min, theta_max
    ("run_a", 2.50, .77, .22, .20, .45, 25, 60),
    ("run_b", 3.00, .85, .18, .20, .55, 10, 60),
    ("run_c", 3.45, .86, .11, .20, .50, 20, 65),
    ("run_d", 4.30, .90, .10, .18, .55, 10, 70),
    ("run_e", 4.45, .88, .09, .20, .55, 15, 70),
]


@pytest.fixture
def run_table(tmp_path):
    path = tmp_path / "runs.csv"
    header = ("run_id,path_length_m,f1_score,hd95_m,radius_xy_m,radius_z_m,"
              "polar_min_deg,polar_max_deg")
    body = "\n".join(",".join(str(v) for v in row) for row in SYNTHETIC_RUNS)
    path.write_text(f"{header}\n{body}\n", encoding="utf-8")
    return path


def test_load_runs_sorts_by_length_and_rejects_malformed(run_table, tmp_path):
    rows = f.load_runs(run_table, f.RUN_COLUMNS + f.GEOMETRY_COLUMNS)
    assert [r["run_id"] for r in rows] == ["run_a", "run_b", "run_c", "run_d", "run_e"]
    for broken in ["run_id,path_length_m\nrun_a,1.0\n",
                   "run_id,path_length_m,f1_score,hd95_m\nrun_a,1.0,1.5,0.1\n",
                   "run_id,path_length_m,f1_score,hd95_m\nrun_a,-1,0.5,0.1\n",
                   "run_id,path_length_m,f1_score,hd95_m\nrun_a,1,x,0.1\n",
                   "run_id,path_length_m,f1_score,hd95_m\nrun_a,1,.5,.1\nrun_a,2,.6,.1\n",
                   "run_id,path_length_m,f1_score,hd95_m\n"]:
        path = tmp_path / "broken.csv"
        path.write_text(broken, encoding="utf-8")
        with pytest.raises(ValidationError): f.load_runs(path)


def test_threshold_selection_returns_shortest_feasible_run(run_table):
    rows = f.load_runs(run_table)
    assert f.select_operating_point(rows, .85, .20)["run_id"] == "run_b"
    assert f.select_operating_point(rows, .85, .12)["run_id"] == "run_c"
    assert f.select_operating_point(rows, .88, .12)["run_id"] == "run_d"
    assert f.select_operating_point(rows, .85, .10)["run_id"] == "run_d"
    assert f.select_operating_point(rows, .99, .001) is None
    with pytest.raises(ValidationError): f.select_operating_point(rows, 2, .1)


def test_multidimensional_dominance_duplicates_and_invalid(run_table):
    costs = [[1,2,3], [1,2,3], [2,2,3], [0,3,3]]
    assert f.nondominated_costs(costs).tolist() == [True,True,False,True]
    with pytest.raises(ValidationError): f.nondominated_costs([[float("nan")]])
    rows = f.load_runs(run_table)
    # run_e is dominated in length/F1 alone, and nondominated once HD95 joins them.
    length_f1 = f.nondominated_costs([[r["path_length_m"], -r["f1_score"]] for r in rows])
    joint = f.nondominated_costs([[r["path_length_m"], -r["f1_score"], r["hd95_m"]] for r in rows])
    assert length_f1.tolist() == [True, True, True, True, False]
    assert joint.all()


def test_weighted_sum_support_excludes_a_nondominated_run(run_table):
    rows = f.load_runs(run_table)
    supported = f.supported_f1_length_points(rows)
    # run_c is nondominated in length/F1 yet sits below the concave envelope, so no
    # nonnegative length penalty can ever make a weighted sum return it.
    assert "run_c" not in supported
    for penalty in np.r_[0, np.logspace(-4, 2, 1000)]:
        winner = max(rows, key=lambda r: r["f1_score"]-penalty*r["path_length_m"])
        assert winner["run_id"] in supported


def test_sightline_inclination_is_below_the_polar_parameter(run_table):
    rows = f.load_runs(run_table, f.RUN_COLUMNS + f.GEOMETRY_COLUMNS)
    geometry = {g["run_id"]: g for g in f.sightline_geometry(rows, focal_height_m=.04)}
    for row in rows:
        g = geometry[row["run_id"]]
        # Every run has r_xy < r_z, so the real sightline is steeper than theta.
        assert g["sightline_polar_min_deg"] < row["polar_min_deg"]
        assert g["sightline_polar_max_deg"] < row["polar_max_deg"]
        assert g["height_min_m"] < g["height_max_m"]
    with pytest.raises(ValidationError):
        f.sightline_geometry([{"run_id": "x"}])
