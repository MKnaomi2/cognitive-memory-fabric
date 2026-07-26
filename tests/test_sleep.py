import hashlib
from dataclasses import replace

import pytest

from hippocampal_memory.circuit import CircuitConfig, TrisynapticCircuit
from hippocampal_memory.coordination import MemoryCoordinator
from hippocampal_memory.sleep import SleepConsolidator
from hippocampal_memory.store import MemoryStore


def tiny_config():
    return replace(
        CircuitConfig(),
        populations={"EC": 128, "DG": 256, "CA3": 128, "CA1": 64},
        fanout={
            "EC_DG": 2,
            "DG_CA3": 2,
            "EC_CA3": 2,
            "CA3_CA3": 2,
            "CA3_CA1": 2,
            "EC_CA1": 2,
            "LOCAL_INHIBITION": 2,
        },
    )


def test_sleep_prioritizes_unbound_then_legacy_memories(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    coordinator = MemoryCoordinator(store)
    current = coordinator.ingest(
        "High salience already encoded memory.",
        actor_type="user",
        salience_score=1.0,
    )
    legacy = coordinator.ingest(
        "Medium salience legacy memory.",
        actor_type="user",
        salience_score=0.5,
    )
    unbound = coordinator.ingest(
        "Low salience memory awaiting neural encoding.",
        actor_type="user",
        salience_score=0.1,
    )
    coordinator.bind_engram(
        current,
        [1],
        circuit_version=CircuitConfig().version,
        encoding_version="content-v3",
        content_sha256=hashlib.sha256(
            "High salience already encoded memory.".encode()
        ).hexdigest(),
        ca1_signature=[10],
    )
    coordinator.bind_engram(
        legacy,
        [2],
        circuit_version="trisynaptic-v2",
        encoding_version="memory-id-v2",
        content_sha256="",
        ca1_signature=[],
    )

    rows = SleepConsolidator(store, device="cpu")._memory_rows(3)

    assert [int(row["fact_id"]) for row in rows] == [unbound, legacy, current]
    store.close()


def test_sleep_resumes_hash_verified_checkpoint(tmp_path):
    pytest.importorskip("torch")
    store = MemoryStore(tmp_path / "memory.db")
    MemoryCoordinator(store)
    config = tiny_config()
    circuit = TrisynapticCircuit(config, device="cpu")
    circuit.stimulate_content("prior sleep state", steps=4, plastic=True)
    checkpoint = circuit.checkpoint(tmp_path / "checkpoint.pt")
    store._conn.execute(
        """
        INSERT INTO neural_checkpoints (
            checkpoint_id,circuit_version,phase,path,sha256,event_revision,
            created_at,metadata_json
        ) VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP,?)
        """,
        (
            "checkpoint-parent",
            config.version,
            "nrem-rem",
            checkpoint["path"],
            checkpoint["sha256"],
            1,
            "{}",
        ),
    )
    loaded, parent = SleepConsolidator(
        store,
        state_root=tmp_path / "neural",
        circuit_config=config,
        device="cpu",
    )._load_circuit()

    assert parent == "checkpoint-parent"
    assert loaded.step_index == circuit.step_index
    assert loaded.pathways[0].weight.equal(circuit.pathways[0].weight)
    store.close()
