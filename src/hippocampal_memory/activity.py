"""Cross-process foreground-turn leases for preemptible local maintenance."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def _lease_root() -> Path:
    configured = os.environ.get("HIPPOCAMPAL_MEMORY_HOME")
    home = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".hippocampal-memory"
    )
    root = home / "runtime" / "foreground"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _last_activity_path() -> Path:
    return _lease_root().parent / "foreground-last-activity.json"


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:
        return pid == os.getpid()


def foreground_active(*, stale_after_seconds: int = 3600) -> bool:
    """Return whether any live foreground turn currently owns a lease."""
    now = time.time()
    for path in _lease_root().glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", 0))
            created_at = float(payload.get("created_at", 0))
            if pid > 0 and _pid_alive(pid) and now - created_at <= stale_after_seconds:
                return True
        except Exception:
            pass
        try:
            path.unlink()
        except OSError:
            pass
    return False


def foreground_idle_seconds() -> float:
    """Return seconds since the most recently completed foreground turn."""
    if foreground_active():
        return 0.0
    try:
        payload = json.loads(_last_activity_path().read_text(encoding="utf-8"))
        return max(0.0, time.time() - float(payload.get("finished_at", time.time())))
    except Exception:
        return float("inf")


@contextmanager
def foreground_turn(session_id: str = ""):
    """Publish one live turn and remove the lease even when the turn raises."""
    lease = _lease_root() / (
        f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}.json"
    )
    payload = {
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "session_id": str(session_id or ""),
        "created_at": time.time(),
    }
    lease.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    try:
        yield
    finally:
        try:
            lease.unlink()
        except OSError:
            pass
        target = _last_activity_path()
        temporary = target.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps({"finished_at": time.time()}, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, target)
        except OSError:
            pass
