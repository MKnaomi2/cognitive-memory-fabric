import hashlib

import pytest

from hippocampal_memory.circuit import TrisynapticCircuit
from hippocampal_memory.coordination import MemoryCoordinator
from hippocampal_memory.neural_service import (
    NeuralReadoutRuntime,
    RemoteNeuralReadout,
    create_neural_readout_app,
)
from hippocampal_memory.readout import ReadoutConfig
from hippocampal_memory.store import MemoryStore


def test_authenticated_neural_service_loads_hashed_checkpoint(tmp_path):
    pytest.importorskip("torch")
    testclient = pytest.importorskip("fastapi.testclient")
    store = MemoryStore(tmp_path / "memory.db")
    coordinator = MemoryCoordinator(store)
    memory_id = coordinator.ingest(
        "Project Atlas uses port 9090.",
        actor_type="user",
        actor_ref="neural-service-test",
    )
    circuit = TrisynapticCircuit(device="cpu")
    query = circuit.query_content("What port does Project Atlas use?")
    coordinator.bind_engram(
        memory_id,
        query["engram_neurons"],
        circuit_version=circuit.config.version,
        encoding_version="content-v3",
        content_sha256=hashlib.sha256(
            "Project Atlas uses port 9090.".encode()
        ).hexdigest(),
        ca1_signature=query["ca1_signature"],
    )
    checkpoint = circuit.checkpoint(tmp_path / "checkpoint.pt")
    store._conn.execute(
        """
        INSERT INTO neural_checkpoints (
            checkpoint_id,circuit_version,phase,path,sha256,event_revision,
            created_at,metadata_json
        ) VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP,?)
        """,
        (
            "checkpoint-test",
            circuit.config.version,
            "awake",
            checkpoint["path"],
            checkpoint["sha256"],
            1,
            "{}",
        ),
    )
    store._conn.commit()

    runtime = NeuralReadoutRuntime(
        store,
        ReadoutConfig(mode="neural", neural_activation_min=0.7),
        device="cpu",
    )
    token = "t" * 64
    client = testclient.TestClient(
        create_neural_readout_app(runtime, token=token),
        client=("127.0.0.1", 50_000),
    )
    assert client.get("/health").status_code == 403
    headers = {"Authorization": f"Bearer {token}"}
    health = client.get("/health", headers=headers)
    assert health.status_code == 200
    assert health.json()["checkpoint_sha256"] == checkpoint["sha256"]

    response = client.post(
        "/rerank",
        headers=headers,
        json={"query": "What port does Project Atlas use?"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["memories"][0]["fact_id"] == memory_id
    assert result["effective_mode"] == "neural"
    assert result["service"]["device"] == "cpu"

    with pytest.raises(ValueError, match="loopback"):
        RemoteNeuralReadout("https://example.com", token)
    store.close()
