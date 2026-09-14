# Contributing

Every behavior-changing patch needs analytic or synthetic tests and a note in the changelog; new measurements require their own experiment manifest. Add optional dependencies behind lazy imports. Never silently drop invalid points, replace empty predictions with dummy geometry, convert failed stages into successes, infer units, or download unspecified model weights.

Use `python -m pytest -q` before proposing a change. Include the tested environment and identify any integration path that was not run. CI is infrastructure, not proof that a robotics experiment was reproduced. Avoid new global state in model/config objects; isolate GPU workers and record exact checkpoints.

Do not commit private data, credentials, full model weights, captured imagery, or generated caches. Changes affecting robot motion require responsible integration review and must not be labelled safe from unit tests alone.
