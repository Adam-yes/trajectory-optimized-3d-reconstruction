# Validation record

Local validation performed for release 0.3.0. It records what was executed and, just as importantly, what was not.

## Executed

**93 tests passed; one optional Open3D triangle-distance test was skipped.** Statement coverage is **87.44%** (856/979 statements) in `src/eyeinhand`, excluding `adapters/`. This is not coverage of the full repository, the GPU integration scripts or the ROS runtime. `ruff check .` reports no findings.

The operating-point analysis is covered by tests that verify every printed candidate rounding, check the recovered grid identifiers against the enumerated grid, and recalculate the finite-set dominance and threshold selections. `tools/render_diagrams.py` regenerates the README overview figure from the trajectory equations and the stage graph.

Geometry and metric tests use analytic or synthetic fixtures. The registration CLI test uses an identity-transform protocol worker and checks file handling and transform composition; it is not a BUFFER-X benchmark. Scene tests validate the configuration and target-object requirement; they do not execute MoveIt collision checking.

## Runtime scope

No VGGT or BUFFER-X GPU inference, no ROS, MoveIt or Isaac Sim end-to-end run, and no physical-robot trial was executed. Exact Open3D triangle distance and the runtime OBB extraction require validation where Open3D is installed. The algebraic OBB transform is CPU-tested.

## Environment

Python 3.13.5. Package versions are in `validation_environment.json`; the recorded run used the pinned `requirements-analysis.txt`. The compatibility ranges in `pyproject.toml` are not a lockfile. The GitHub Actions workflow is supplied but no remote CI run is claimed.
