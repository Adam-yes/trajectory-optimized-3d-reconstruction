# Security and operational boundaries

This is research software, not a safety controller. Never run unknown checkpoints through unrestricted pickle loaders; the VGGT adapter uses `weights_only=True` and verifies a local SHA-256. Model factories and subprocess commands execute trusted local code and must be reviewed before use. Do not execute untrusted factories or extracted archives.

The process wrapper never invokes a shell and only terminates its own process group. This does not sandbox a malicious executable. Run external ML stacks with least privilege, read-only inputs and isolated output directories. A stale run lease is a reason to inspect a prior process, not to kill unrelated processes.

No vulnerability-report email or security response time is invented. The repository owner must configure a private reporting route before a public release.
