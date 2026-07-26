"""Executable real-backend integration coverage kept outside the 48-unit-test gate."""

from __future__ import annotations

import tempfile
from pathlib import Path

import msgpack
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from observatory_fixture import E2E_TOKEN, build_fixture


def run() -> None:
    root = Path(tempfile.mkdtemp(prefix="cmf-observatory-integration-"))
    fixture = build_fixture(root)
    app = fixture.app()
    try:
        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["geometry_schema"] == 2

            geometry = client.get("/geometry")
            assert geometry.status_code == 200
            assert geometry.json()["neuron_count"] == 36_864
            assert sum(
                pathway["synapse_count"]
                for pathway in geometry.json()["pathways"]
            ) == 770_048

            neuron = client.get("/neuron/0")
            assert neuron.status_code == 200
            assert neuron.json()["neuron_id"] == 0
            assert "incoming" in neuron.json() and "outgoing" in neuron.json()

            frame = {
                "step": 4242,
                "phase": "nrem",
                "active_neurons": [0, 1, 2],
                "active_edges": [[0, 1, 0.5, "EC_DG"]],
                "region_spikes": {"EC": 1, "DG": 1, "CA3": 1, "CA1": 0},
            }
            payload = msgpack.packb(frame, use_bin_type=True)
            assert client.post("/ingest", content=payload).status_code == 403
            accepted = client.post(
                "/ingest",
                content=payload,
                headers={
                    "authorization": f"Bearer {E2E_TOKEN}",
                    "content-type": "application/msgpack",
                },
            )
            assert accepted.status_code == 200

            malformed = client.post(
                "/ingest",
                content=b"not-messagepack",
                headers={
                    "authorization": f"Bearer {E2E_TOKEN}",
                    "content-type": "application/msgpack",
                },
            )
            assert malformed.status_code == 400
            oversized = client.post(
                "/ingest",
                content=b"x",
                headers={
                    "authorization": f"Bearer {E2E_TOKEN}",
                    "content-length": str(16 * 1024 * 1024 + 1),
                },
            )
            assert oversized.status_code == 413

            with client.websocket_connect(
                "/live", headers={"origin": "http://127.0.0.1:5173"}
            ) as websocket:
                assert msgpack.unpackb(websocket.receive_bytes(), raw=False) == frame
            try:
                with client.websocket_connect(
                    "/live", headers={"origin": "https://non-loopback.invalid"}
                ) as websocket:
                    websocket.receive_bytes()
            except WebSocketDisconnect as exc:
                assert exc.code == 1008
            else:
                raise AssertionError("non-loopback WebSocket origin was accepted")

            recordings = client.get("/recordings").json()["recordings"]
            assert any(item["name"] == fixture.recording.name for item in recordings)
            downloaded = client.get(f"/recordings/{fixture.recording.name}")
            assert downloaded.status_code == 200
            assert downloaded.content == fixture.recording.read_bytes()
            traversal = client.get("/recordings/..%5Csecret.hmrec")
            assert traversal.status_code == 400
    finally:
        fixture.store.close()


if __name__ == "__main__":
    run()
    print("observatory integration: PASS")
