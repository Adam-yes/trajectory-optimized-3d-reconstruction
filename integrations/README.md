# Integration qualification boundary

`bufferx_worker.py` implements the source snapshot's collator/model-call protocol while fixing checkpoint identity, mutable config sharing and sampling seed control. It needs a trusted offline model factory for the exact installed package. Supply that package, its checkpoint and the factory yourself; the dependency is stated explicitly rather than hidden behind an automatic download.

`ros2_capture.py` supplies checked action completion and fresh-image/time-aligned TF capture helpers. It intentionally does not recreate a MoveIt planning scene, reset the robot, command an unchecked home move, weaken dynamics, or declare a trajectory safe. Integrate it into a ROS workspace only after scene, calibration and controller qualification. It is source-checked and has not been run against ROS or Isaac Sim here.

Neither helper is part of the passing CPU reconstruction experiment count. There is no such newly executed experiment count in this release. The repository owner must run and record GPU/robotics integration tests before calling these paths validated.
