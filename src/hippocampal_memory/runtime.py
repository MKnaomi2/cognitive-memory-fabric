"""Exclusive local-GPU sleep windows with rapid foreground preemption."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.request
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class GPUStatus:
    used_mib: int
    total_mib: int
    utilization_percent: int


def gpu_status() -> GPUStatus | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).strip()
        used, total, utilization = (int(value.strip()) for value in output.split(","))
        return GPUStatus(used, total, utilization)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def unload_ollama(model: str, endpoint: str = "http://127.0.0.1:11434") -> bool:
    """Ask loopback Ollama to release a model; it reloads on the next request."""
    body = json.dumps({"model": model, "keep_alive": 0}).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read(64)
        return True
    except OSError:
        return False


class ExclusiveSleepWindow(AbstractContextManager["ExclusiveSleepWindow"]):
    """Own the GPU for replay and expose a sub-two-second preemption signal."""

    _lock = threading.Lock()

    def __init__(
        self,
        *,
        model: str,
        foreground_active: Callable[[], bool],
        poll_seconds: float = 0.5,
        max_initial_gpu_mib: int = 1024,
    ) -> None:
        self.model = model
        self.foreground_active = foreground_active
        self.poll_seconds = min(1.0, max(0.1, poll_seconds))
        self.max_initial_gpu_mib = max_initial_gpu_mib
        self._stop = threading.Event()
        self._preempt = threading.Event()
        self._watcher: threading.Thread | None = None

    def __enter__(self) -> "ExclusiveSleepWindow":
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("another neural sleep window is already active")
        if self.foreground_active():
            self._lock.release()
            raise RuntimeError("foreground work is active")
        unload_ollama(self.model)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            status = gpu_status()
            if status is None or status.used_mib <= self.max_initial_gpu_mib:
                break
            if self.foreground_active():
                self._lock.release()
                raise RuntimeError("foreground work started while acquiring GPU")
            time.sleep(0.25)
        self._watcher = threading.Thread(target=self._watch, daemon=True)
        self._watcher.start()
        return self

    def _watch(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            if self.foreground_active():
                self._preempt.set()
                return

    def should_preempt(self) -> bool:
        return self._preempt.is_set() or self.foreground_active()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._watcher:
            self._watcher.join(timeout=1.2)
        self._lock.release()
