"""Persistent authenticated CUDA readout service for thin Hermes runtimes."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .circuit import TrisynapticCircuit
from .readout import MemoryReadout, ReadoutConfig
from .store import MemoryStore
from .telemetry import observatory_token

try:
    from fastapi import FastAPI, HTTPException, Request
except ImportError:  # pragma: no cover - optional dependency check.
    FastAPI = None  # type: ignore[assignment,misc]

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_MAX_QUERY_CHARS = 16_000
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def _require_loopback_url(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in _LOOPBACK_HOSTS:
        raise ValueError("neural readout endpoint must use loopback HTTP")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("neural readout endpoint contains unsupported URL fields")
    return endpoint.rstrip("/")


class RemoteNeuralReadout:
    """Bounded client used by Hermes when Torch lives in another environment."""

    def __init__(
        self,
        endpoint: str,
        token: str = "",
        *,
        token_path: str | Path | None = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        self.endpoint = _require_loopback_url(endpoint)
        if token and len(token) < 64:
            raise ValueError("neural readout token is too short")
        if not token and token_path is None:
            raise ValueError("neural readout token or token path is required")
        self.token = token
        self.token_path = Path(token_path) if token_path is not None else None
        self.timeout_seconds = max(0.1, min(float(timeout_seconds), 10.0))

    @classmethod
    def for_store(
        cls,
        store: MemoryStore,
        endpoint: str,
        *,
        timeout_seconds: float = 2.0,
    ) -> RemoteNeuralReadout:
        return cls(
            endpoint,
            token_path=store.db_path.parent / "runtime" / "neural-readout.token",
            timeout_seconds=timeout_seconds,
        )

    def _token(self) -> str:
        if self.token:
            return self.token
        assert self.token_path is not None
        token = observatory_token(self.token_path)
        if token is None:
            raise RuntimeError("neural readout token is unavailable")
        return token

    def search(
        self, query: str, *, semantic_vector: list[float] | None = None
    ) -> dict[str, Any]:
        payload = json.dumps(
            {"query": query, "semantic_vector": semantic_vector},
            separators=(",", ":"),
        ).encode()
        if len(payload) > _MAX_REQUEST_BYTES:
            raise ValueError("neural readout request exceeds bound")
        request = urllib.request.Request(
            self.endpoint + "/rerank",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._token()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                body = response.read(_MAX_RESPONSE_BYTES + 1)
        except (OSError, urllib.error.HTTPError) as exc:
            raise RuntimeError(f"neural readout service unavailable: {exc}") from exc
        if len(body) > _MAX_RESPONSE_BYTES:
            raise RuntimeError("neural readout response exceeds bound")
        result = json.loads(body)
        if not isinstance(result, dict) or not isinstance(
            result.get("memories"), list
        ):
            raise RuntimeError("neural readout response is invalid")
        return result


class NeuralReadoutRuntime:
    """Own one CUDA circuit and reload a newly registered checkpoint by hash."""

    def __init__(
        self,
        store: MemoryStore,
        config: ReadoutConfig,
        *,
        device: str = "cuda",
    ) -> None:
        self.store = store
        self.config = config
        self.device = device
        self._lock = threading.Lock()
        self._checkpoint_id = ""
        self._checkpoint_sha256 = ""
        self._readout: MemoryReadout | None = None

    def _latest_checkpoint(self) -> dict[str, str]:
        row = self.store._conn.execute(
            """
            SELECT checkpoint_id,path,sha256 FROM neural_checkpoints
            WHERE circuit_version='trisynaptic-v3-content-readout'
            ORDER BY created_at DESC, rowid DESC LIMIT 1
            """
        ).fetchone()
        if row is None:
            raise RuntimeError("no compatible neural checkpoint is registered")
        return {key: str(row[key]) for key in ("checkpoint_id", "path", "sha256")}

    def _ensure_readout(self) -> None:
        checkpoint = self._latest_checkpoint()
        if (
            self._readout is not None
            and checkpoint["checkpoint_id"] == self._checkpoint_id
        ):
            return
        path = Path(checkpoint["path"]).resolve()
        if not path.is_file():
            raise RuntimeError("registered neural checkpoint is missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not hmac.compare_digest(digest, checkpoint["sha256"]):
            raise RuntimeError("registered neural checkpoint hash mismatch")
        circuit = TrisynapticCircuit.from_checkpoint(path, device=self.device)
        self._readout = MemoryReadout(self.store, self.config, circuit=circuit)
        self._checkpoint_id = checkpoint["checkpoint_id"]
        self._checkpoint_sha256 = digest

    def search(
        self, query: str, *, semantic_vector: list[float] | None = None
    ) -> dict[str, Any]:
        if not query.strip() or len(query) > _MAX_QUERY_CHARS:
            raise ValueError("query is empty or exceeds bound")
        with self._lock:
            self._ensure_readout()
            assert self._readout is not None
            result = self._readout.search(query, semantic_vector=semantic_vector)
            result["service"] = {
                "checkpoint_id": self._checkpoint_id,
                "checkpoint_sha256": self._checkpoint_sha256,
                "device": self.device,
            }
            return result

    def health(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_readout()
            assert self._readout is not None
            circuit = self._readout.circuit
            return {
                "status": "ok",
                "device": str(circuit.device if circuit else self.device),
                "checkpoint_id": self._checkpoint_id,
                "checkpoint_sha256": self._checkpoint_sha256,
            }


def create_neural_readout_app(
    runtime: NeuralReadoutRuntime,
    *,
    token: str,
) -> Any:
    """Create an authenticated loopback-only neural inference API."""
    if FastAPI is None:
        raise RuntimeError("install the 'observatory' extra")
    if len(token) < 64:
        raise ValueError("neural readout token is missing or too short")
    app = FastAPI(title="Cognitive Memory Neural Readout", docs_url=None, redoc_url=None)

    def authorize(request: Request) -> None:
        client_host = request.client.host if request.client else ""
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
        if client_host not in _LOOPBACK_HOSTS or not hmac.compare_digest(
            supplied, token
        ):
            raise HTTPException(403, "neural readout authentication failed")

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        authorize(request)
        try:
            return await asyncio.to_thread(runtime.health)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/rerank")
    async def rerank(request: Request) -> dict[str, Any]:
        authorize(request)
        try:
            length = int(request.headers.get("content-length", "0") or "0")
        except ValueError as exc:
            raise HTTPException(400, "invalid content length") from exc
        if length <= 0 or length > _MAX_REQUEST_BYTES:
            raise HTTPException(413, "neural readout request exceeds bound")
        body = await request.body()
        if len(body) != length:
            raise HTTPException(400, "neural readout request length mismatch")
        try:
            payload = json.loads(body)
            query = str(payload.get("query") or "")
            semantic_vector = payload.get("semantic_vector")
            if semantic_vector is not None and not isinstance(semantic_vector, list):
                raise ValueError("semantic_vector must be a list")
            return await asyncio.to_thread(
                runtime.search, query, semantic_vector=semantic_vector
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    return app
