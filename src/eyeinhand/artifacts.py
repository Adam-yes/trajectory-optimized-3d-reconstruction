"""Strict JSON, content-addressed inputs, atomic writes and explicit run leases."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator

from . import __version__
from .errors import ArtifactConflictError, ValidationError


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Not strict JSON: {exc}") from exc


def _unique_pairs(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValidationError(f"Nonstandard JSON numeric constant: {value}")


def read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValidationError(f"Invalid JSON in {path}: {exc}") from exc


def atomic_json(path: str | Path, value: Any) -> None:
    """Replace only after a complete, fsynced strict-JSON document exists."""
    raw = canonical_json(value) + b"\n"
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=f".{dest.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, dest)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def fingerprint(config: dict, files: dict[str, str | Path], *, software: dict) -> str:
    """Paths are logical names; their contents and the explicit software identity matter."""
    content = {name: sha256_file(path) for name, path in sorted(files.items())}
    return hashlib.sha256(canonical_json({"schema": "eyeinhand.cache/2", "config": config,
                                         "inputs": content, "software": software})).hexdigest()


@contextmanager
def run_lease(directory: str | Path) -> Iterator[Path]:
    """Fail rather than share a run directory. Stale leases require manual inspection."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ArtifactConflictError(f"Run is locked: {lock}; inspect before removing a stale lease") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid()}))
        yield root
    finally:
        lock.unlink(missing_ok=True)


def completed_artifacts_match(directory: str | Path, expected_fingerprint: str) -> bool:
    """A filename alone never means success; verify state, provenance and all outputs."""
    root = Path(directory).resolve()
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return False
    data = read_json(manifest)
    if (data.get("state") != "complete" or data.get("fingerprint") != expected_fingerprint
            or not data.get("artifacts")):
        return False
    for rel, digest in data["artifacts"].items():
        output = (root / rel).resolve()
        if not output.is_relative_to(root) or not output.is_file() or sha256_file(output) != digest:
            return False
    return True


def environment_manifest() -> dict:
    packages = {}
    for name in ("numpy", "scipy", "Pillow", "torch", "open3d", "trimesh", "scikit-learn", "pytest"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    source_root = Path(__file__).resolve().parent
    source_digest = hashlib.sha256()
    for source_file in sorted(source_root.rglob("*.py")):
        source_digest.update(str(source_file.relative_to(source_root)).encode("utf-8"))
        source_digest.update(source_file.read_bytes())
    return {"eyeinhand": __version__, "core_source_sha256": source_digest.hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(), "packages": packages}
