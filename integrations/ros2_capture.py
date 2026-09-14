"""Optional ROS 2 collection helpers; no ROS imports occur at module import time.

These helpers are NOT a validated robot controller or a complete replacement for
MoveIt planning/scene setup. They check action completion and timestamped capture
instead of turning missing TF or failed writes into successful observations.
Integration must be qualified against the exact ROS/MoveIt/Isaac installation.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

from eyeinhand.artifacts import atomic_json, sha256_file
from eyeinhand.errors import StageError, ValidationError


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def wait_future(node, future, timeout_s: float):
    import rclpy
    deadline = time.monotonic() + timeout_s
    while rclpy.ok() and not future.done():
        if time.monotonic() >= deadline:
            raise StageError('ROS future timed out')
        rclpy.spin_once(node, timeout_sec=min(.05, max(0., deadline-time.monotonic())))
    if not future.done():
        raise StageError('ROS shutdown before future completion')
    if future.exception() is not None:
        raise StageError('ROS future raised an exception') from future.exception()
    result = future.result()
    if result is None:
        raise StageError('ROS future returned no result')
    return result


def execute_checked(node, action_client, trajectory, *, simulation_acknowledged: bool,
                    timeout_s: float = 120.) -> dict:
    """Send only an externally planned/retimed trajectory; never synthesize a home jump."""
    from action_msgs.msg import GoalStatus
    from control_msgs.action import FollowJointTrajectory
    if not simulation_acknowledged or not node.get_parameter('use_sim_time').value:
        raise ValidationError('This adapter requires explicit simulation acknowledgement and use_sim_time')
    if not trajectory.points or not trajectory.joint_names:
        raise ValidationError('Empty trajectory')
    if not action_client.wait_for_server(timeout_sec=5.):
        raise StageError('Trajectory action server unavailable')
    goal = FollowJointTrajectory.Goal()
    goal.trajectory = trajectory
    handle = wait_future(node, action_client.send_goal_async(goal), 10.)
    if not handle.accepted:
        raise StageError('Trajectory action was rejected')
    try:
        wrapped = wait_future(node, handle.get_result_async(), timeout_s)
    except BaseException:
        # Request cancellation; do not claim it has physically stopped unless confirmed.
        handle.cancel_goal_async()
        raise
    result = wrapped.result
    success = (wrapped.status == GoalStatus.STATUS_SUCCEEDED
               and result.error_code == FollowJointTrajectory.Result.SUCCESSFUL)
    if not success:
        raise StageError(f'Trajectory failed: status={wrapped.status}, code={result.error_code}, '
                         f'message={result.error_string}')
    return {'action_succeeded': True, 'action_status': wrapped.status,
            'action_error_code': result.error_code, 'collision_safety_certified': False}


def capture_after(node, tf_buffer, latest_image: Callable, *, after_stamp_ns: int,
                  world_frame: str, camera_frame: str, flange_frame: str,
                  output: str | Path, timeout_s: float = 5.) -> dict:
    """Save a fresh image only if camera and flange TF exist at that image timestamp.

    latest_image() must return the most recent ROS sensor_msgs/Image or None.
    The caller owns its subscriber and executor; this function spins a single node.
    """
    import cv2
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.time import Time
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / 'image.png'
    metadata_path = root / 'capture.json'
    if image_path.exists() or metadata_path.exists():
        raise ValidationError('Capture output already exists')
    deadline = time.monotonic() + timeout_s
    message = None
    while time.monotonic() < deadline:
        candidate = latest_image()
        if candidate is not None and stamp_ns(candidate.header.stamp) > after_stamp_ns:
            message = candidate
            break
        rclpy.spin_once(node, timeout_sec=.05)
    if message is None:
        raise StageError('No fresh post-motion image was received')
    timestamp = Time.from_msg(message.header.stamp)
    transforms = {}
    for frame in (camera_frame, flange_frame):
        try:
            tf = tf_buffer.lookup_transform(world_frame, frame, timestamp)
        except Exception as exc:
            raise StageError(f'Missing synchronized TF for {frame}; no fabricated pose is saved') from exc
        t, q = tf.transform.translation, tf.transform.rotation
        transforms[frame] = {'translation_m': [t.x, t.y, t.z],
                             'quaternion_xyzw': [q.x, q.y, q.z, q.w]}
    image = CvBridge().imgmsg_to_cv2(message, desired_encoding='bgr8')
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise StageError('Image encoding failed')
    temporary = root / '.image.png.tmp'
    try:
        with temporary.open('xb') as stream:
            stream.write(encoded.tobytes()); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, image_path)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {'schema': 'eyeinhand.capture/2', 'image_stamp_ns': stamp_ns(message.header.stamp),
                'world_frame': world_frame, 'image_frame_id': message.header.frame_id,
                'transforms': transforms, 'image_sha256': sha256_file(image_path),
                'fresh_after_motion': True}
    atomic_json(metadata_path, metadata)
    return metadata
