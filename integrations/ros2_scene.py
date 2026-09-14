"""Synchronous MoveIt scene setup for the simulated workcell.

No robot motion is commanded. Source dimensions are in the base_link frame.
The central target-object exclusion volume is mandatory. Verify these dimensions
against the loaded simulation before using a different scene or object arrangement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eyeinhand.artifacts import read_json
from eyeinhand.errors import StageError, ValidationError


def validate_scene(spec: dict) -> None:
    if set(spec) != {"world_frame", "attachment_link", "world_boxes", "attached_boxes"}:
        raise ValidationError("Unexpected scene fields")
    if not spec["world_frame"] or not spec["attachment_link"]:
        raise ValidationError("Scene frame names are required")
    ids = [box["id"] for box in spec["world_boxes"]]
    if len(ids) != len(set(ids)) or "target_objects" not in ids or "table_surface" not in ids:
        raise ValidationError("Unique world IDs, table and target_objects exclusion volume are required")
    if len(spec["attached_boxes"]) != 3:
        raise ValidationError("Expected the module, fingers and camera protection boxes")
    for box in [*spec["world_boxes"], *spec["attached_boxes"]]:
        d, p = np.asarray(box["size_m"], float), np.asarray(box["center_m"], float)
        if d.shape != (3,) or p.shape != (3,) or not np.isfinite([d, p]).all() or (d <= 0).any():
            raise ValidationError("Collision boxes require finite centers and positive 3D dimensions")


def apply_scene(node, spec: dict, *, timeout_s: float = 10.) -> dict:
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
    from moveit_msgs.srv import ApplyPlanningScene
    from ros2_capture import wait_future
    from shape_msgs.msg import SolidPrimitive

    validate_scene(spec)
    if not node.get_parameter("use_sim_time").value:
        raise ValidationError("The scene adapter requires use_sim_time=True")
    client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
    if not client.wait_for_service(timeout_sec=timeout_s):
        raise StageError("MoveIt apply_planning_scene service unavailable")
    request = ApplyPlanningScene.Request()
    request.scene.is_diff = True
    request.scene.robot_state.is_diff = True

    def add_box(obj, box):
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = list(map(float, box["size_m"]))
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, box["center_m"])
        pose.orientation.w = 1.0  # A zero quaternion is not a valid box orientation.
        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose)

    for box in spec["world_boxes"]:
        obj = CollisionObject()
        obj.header.frame_id, obj.id = spec["world_frame"], box["id"]
        obj.operation = CollisionObject.ADD
        add_box(obj, box)
        request.scene.world.collision_objects.append(obj)
    attached = AttachedCollisionObject()
    attached.link_name = attached.object.header.frame_id = spec["attachment_link"]
    attached.object.id = "gripper_camera_assembly"
    attached.object.operation = CollisionObject.ADD
    # Only the actual mounting link is allowed to touch the attached assembly.
    # Target objects are world obstacles and are never touch links.
    attached.touch_links = [spec["attachment_link"]]
    for box in spec["attached_boxes"]:
        add_box(attached.object, box)
    request.scene.robot_state.attached_collision_objects.append(attached)
    response = wait_future(node, client.call_async(request), timeout_s)
    if not response.success:
        raise StageError("MoveIt rejected the scene update")
    return {"applied": True, "world_ids": [b["id"] for b in spec["world_boxes"]],
            "target_object_collision_checking": "world obstacle; no collision exemptions added",
            "attached_boxes": 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--simulation", action="store_true", required=True)
    args = parser.parse_args()
    import rclpy
    from rclpy.parameter import Parameter
    rclpy.init()
    node = rclpy.create_node("eyeinhand_scene", parameter_overrides=[Parameter("use_sim_time", value=True)])
    try:
        print(apply_scene(node, read_json(args.config)))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
