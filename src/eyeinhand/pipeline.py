"""Bounded in-process orchestration and isolated external command execution."""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TypeVar

from .errors import StageError, ValidationError

T = TypeVar("T")
R = TypeVar("R")


def process_bounded(items: Iterable[T], consumer: Callable[[T], R], *, queue_size: int = 2) -> list[R]:
    """Propagate failures; never wait forever for a producer that forgot its sentinel.

    One consumer preserves GPU serialization. Cancellation is cooperative between
    items, not a way to interrupt a hung GPU kernel; use the subprocess boundary for
    hard deadlines. Results preserve input order.
    """
    if not isinstance(queue_size, int) or isinstance(queue_size, bool) or queue_size < 1:
        raise ValidationError("queue_size must be a positive integer")
    work: queue.Queue = queue.Queue(queue_size)
    stop = threading.Event()
    finished = threading.Event()
    errors: queue.Queue = queue.Queue()

    def produce() -> None:
        try:
            for item in items:
                while not stop.is_set():
                    try:
                        work.put(item, timeout=0.05)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    break
        except BaseException as exc:
            errors.put(exc)
        finally:
            finished.set()

    producer = threading.Thread(target=produce, name="eyeinhand-producer", daemon=True)
    producer.start()
    result = []
    try:
        while True:
            if not errors.empty():
                raise StageError("Producer failed") from errors.get()
            try:
                item = work.get(timeout=0.05)
            except queue.Empty:
                if finished.is_set():
                    break
                continue
            result.append(consumer(item))
    finally:
        stop.set()
        producer.join(timeout=2)
    if producer.is_alive():
        raise StageError("Producer did not stop; input iterator may block indefinitely")
    return result


def run_command(argv: Sequence[str], *, cwd: str | Path, log_path: str | Path,
                timeout_s: float, env: dict[str, str] | None = None) -> dict:
    """No shell interpolation, no global pkill; timeout affects only this process group."""
    if isinstance(argv, (str, bytes)) or not argv or not all(isinstance(s, str) for s in argv):
        raise ValidationError("Command must be a nonempty sequence of strings")
    if timeout_s <= 0:
        raise ValidationError("timeout_s must be positive")
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stream:
        child = subprocess.Popen(list(argv), cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=(os.name == "posix"))
        try:
            code = child.wait(timeout=timeout_s)
        except (subprocess.TimeoutExpired, BaseException) as exc:
            # Kill only this invocation and children that retain its process group.
            if child.poll() is None:
                if os.name == "posix":
                    os.killpg(child.pid, signal.SIGTERM)
                else:
                    child.terminate()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(child.pid, signal.SIGKILL)
                    else:
                        child.kill()
                    child.wait()
            if isinstance(exc, subprocess.TimeoutExpired):
                raise StageError(f"Stage exceeded {timeout_s:g}s; log: {log}") from exc
            raise
    if code != 0:
        raise StageError(f"Stage exited with code {code}; log: {log}")
    return {"returncode": code, "argv": list(argv), "log_path": str(log)}
