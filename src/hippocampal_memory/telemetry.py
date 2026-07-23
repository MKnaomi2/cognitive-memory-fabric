"""Loopback-only, read-only neural observatory telemetry."""

from __future__ import annotations

import asyncio
import hmac
import json
import re
import secrets
import struct
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Iterable

from .coordination import MemoryCoordinator
from .store import MemoryStore

try:
    import msgpack
    from fastapi import (
        FastAPI,
        HTTPException,
        Request,
        WebSocket,
        WebSocketDisconnect,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
except ImportError:  # pragma: no cover - exercised by optional dependency check.
    msgpack = None  # type: ignore[assignment]
    FastAPI = None  # type: ignore[assignment,misc]

_SAFE_RECORDING = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*\.hmrec$")
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_ORIGINS = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]


class TelemetryHub:
    """Thread-safe bridge from simulation workers to async WebSockets."""

    def __init__(self, history_frames: int = 600) -> None:
        self.history: deque[bytes] = deque(maxlen=history_frames)
        self.clients: set[Any] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def encode(self, frame: dict[str, Any]) -> bytes:
        if msgpack is None:
            raise RuntimeError("install the 'observatory' extra")
        return msgpack.packb(frame, use_bin_type=True)

    async def publish(self, frame: dict[str, Any]) -> None:
        payload = self.encode(frame)
        with self._lock:
            self.history.append(payload)
        stale = []
        for client in tuple(self.clients):
            try:
                await client.send_bytes(payload)
            except Exception:
                stale.append(client)
        for client in stale:
            self.clients.discard(client)

    def publish_from_thread(self, frame: dict[str, Any]) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.publish(frame), self._loop)


class FrameRecorder:
    """Length-prefixed MessagePack recording with explicit size bounds."""

    MAGIC = b"HMREC1\n"

    def __init__(self, path: str | Path, max_bytes: int = 512 * 1024 * 1024) -> None:
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("xb")
        self._stream.write(self.MAGIC)

    def append(self, payload: bytes) -> None:
        if self._stream.tell() + len(payload) + 4 > self.max_bytes:
            raise RuntimeError("recording reached configured size bound")
        self._stream.write(struct.pack("<I", len(payload)))
        self._stream.write(payload)

    def close(self) -> None:
        self._stream.flush()
        self._stream.close()

    def __enter__(self) -> "FrameRecorder":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def read_recording(path: str | Path) -> Iterable[bytes]:
    with Path(path).open("rb") as stream:
        if stream.read(len(FrameRecorder.MAGIC)) != FrameRecorder.MAGIC:
            raise ValueError("invalid observatory recording")
        while True:
            header = stream.read(4)
            if not header:
                return
            if len(header) != 4:
                raise ValueError("truncated frame header")
            length = struct.unpack("<I", header)[0]
            if length > 16 * 1024 * 1024:
                raise ValueError("frame exceeds safety bound")
            payload = stream.read(length)
            if len(payload) != length:
                raise ValueError("truncated frame")
            yield payload


def observatory_token(path: str | Path, *, create: bool = False) -> str | None:
    target = Path(path)
    if target.is_file():
        value = target.read_text(encoding="utf-8").strip()
        return value if len(value) >= 64 else None
    if not create:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_hex(32)
    target.write_text(value, encoding="utf-8")
    return value


class RemoteTelemetry:
    """Authenticated loopback publisher used by a separate sleep process."""

    def __init__(
        self,
        token: str,
        endpoint: str = "http://127.0.0.1:8765/ingest",
    ) -> None:
        self.token = token
        self.endpoint = endpoint

    @staticmethod
    def encode(frame: dict[str, Any]) -> bytes:
        if msgpack is None:
            raise RuntimeError("install the 'observatory' extra")
        return msgpack.packb(frame, use_bin_type=True)

    def publish_from_thread(self, frame: dict[str, Any]) -> None:
        payload = self.encode(frame)
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/msgpack",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=1.5) as response:
                response.read(32)
        except OSError:
            # Recording remains authoritative when no viewer is running.
            return


def create_app(
    store: MemoryStore,
    *,
    geometry: dict[str, Any] | None = None,
    hub: TelemetryHub | None = None,
    recordings_root: str | Path | None = None,
    publisher_token: str | None = None,
) -> Any:
    """Create a viewer API with no memory-lifecycle write endpoints."""
    if FastAPI is None or msgpack is None:
        raise RuntimeError("install the 'observatory' extra")
    MemoryCoordinator(store)  # Ensure read projections exist for fresh databases.
    hub = hub or TelemetryHub()
    recordings = Path(recordings_root or store.db_path.parent / "recordings").resolve()
    recordings.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Hippocampal Neural Observatory", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup() -> None:
        hub.attach_loop(asyncio.get_running_loop())

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "transport": "loopback", "schema": 1}

    @app.get("/geometry")
    async def circuit_geometry() -> dict[str, Any]:
        if geometry is None:
            raise HTTPException(503, "circuit geometry is not loaded")
        return geometry

    @app.get("/snapshot")
    async def snapshot() -> dict[str, Any]:
        with store._lock:
            facts = store._conn.execute(
                """
                SELECT status, memory_kind, COUNT(*) count,
                       AVG(trust_score) confidence
                FROM facts GROUP BY status, memory_kind
                """
            ).fetchall()
            conflicts = store._conn.execute(
                "SELECT COUNT(*) FROM fact_conflicts WHERE status='open'"
            ).fetchone()[0]
            events = store._conn.execute(
                "SELECT COUNT(*) FROM memory_events"
            ).fetchone()[0]
        return {
            "memories": [dict(row) for row in facts],
            "open_conflicts": int(conflicts),
            "event_count": int(events),
            "server_time": time.time(),
        }

    @app.get("/memory/{memory_id}")
    async def memory(memory_id: int) -> dict[str, Any]:
        with store._lock:
            row = store._conn.execute(
                """
                SELECT f.*, b.engram_id, b.neuron_ids_json
                FROM facts f LEFT JOIN engram_bindings b
                  ON b.memory_id=CAST(f.fact_id AS TEXT)
                WHERE f.fact_id=?
                """,
                (memory_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(404, "memory not found")
            evidence = store._conn.execute(
                """
                SELECT polarity, source_type, source_ref, weight, observed_at
                FROM fact_evidence WHERE fact_id=? ORDER BY observed_at
                """,
                (memory_id,),
            ).fetchall()
            events = store._conn.execute(
                """
                SELECT event_type, revision, occurred_at, actor_type, actor_ref,
                       source_uri, payload_sha256
                FROM memory_events WHERE aggregate_id=?
                ORDER BY revision
                """,
                (f"memory:{memory_id}",),
            ).fetchall()
        result = dict(row)
        # The observatory deliberately exposes provenance and summaries, never
        # raw transcript content or hidden model reasoning.
        result.pop("provenance_json", None)
        result.pop("hrr_vector", None)
        result["evidence"] = [dict(item) for item in evidence]
        result["events"] = [dict(item) for item in events]
        if result.get("neuron_ids_json"):
            result["neuron_ids"] = json.loads(result.pop("neuron_ids_json"))
        return result

    @app.post("/ingest")
    async def ingest(request: Request) -> dict[str, bool]:
        client_host = request.client.host if request.client else ""
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
        if (
            client_host not in _LOOPBACK_HOSTS
            or not publisher_token
            or not hmac.compare_digest(supplied, publisher_token)
        ):
            raise HTTPException(403, "publisher authentication failed")
        content_length = int(request.headers.get("content-length", "0") or "0")
        if content_length <= 0 or content_length > 16 * 1024 * 1024:
            raise HTTPException(413, "telemetry frame exceeds bound")
        payload = await request.body()
        frame = msgpack.unpackb(payload, raw=False)
        if not isinstance(frame, dict) or "step" not in frame:
            raise HTTPException(400, "invalid telemetry frame")
        await hub.publish(frame)
        return {"accepted": True}

    @app.get("/recordings")
    async def list_recordings() -> dict[str, Any]:
        return {
            "recordings": [
                {"name": item.name, "bytes": item.stat().st_size}
                for item in sorted(recordings.glob("*.hmrec"))
                if item.is_file()
            ]
        }

    @app.get("/recordings/{name}")
    async def recording(name: str) -> Any:
        if not _SAFE_RECORDING.fullmatch(name):
            raise HTTPException(400, "invalid recording name")
        path = (recordings / name).resolve()
        try:
            path.relative_to(recordings)
        except ValueError as exc:
            raise HTTPException(400, "path traversal rejected") from exc
        if not path.is_file():
            raise HTTPException(404, "recording not found")
        return FileResponse(path, media_type="application/octet-stream")

    @app.websocket("/live")
    async def live(websocket: WebSocket) -> None:
        client_host = websocket.client.host if websocket.client else ""
        origin = websocket.headers.get("origin")
        if client_host not in _LOOPBACK_HOSTS or (origin and origin not in _ORIGINS):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        hub.clients.add(websocket)
        try:
            with hub._lock:
                backlog = list(hub.history)
            for payload in backlog:
                await websocket.send_bytes(payload)
            while True:
                # Incoming data is ignored; this is a read-only channel.
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(websocket)

    return app


def serve(app: Any, host: str = "127.0.0.1", port: int = 8765) -> None:
    if host not in {"127.0.0.1", "::1"}:
        raise ValueError("the observatory may bind only to loopback")
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
