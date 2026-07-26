import hashlib
from dataclasses import replace

import pytest

import hippocampal_memory.sleep as sleep_module
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


def test_sleep_prioritizes_pending_then_unbound_then_legacy_memories(tmp_path):
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
    pending = coordinator.ingest(
        "Interrupted encoding awaiting checkpoint recovery.",
        actor_type="user",
        salience_score=0.05,
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
    coordinator.bind_engram(
        pending,
        [3],
        circuit_version=CircuitConfig().version,
        encoding_version="content-v3-pending:failed-session",
        content_sha256=hashlib.sha256(
            "Interrupted encoding awaiting checkpoint recovery.".encode()
        ).hexdigest(),
        ca1_signature=[11],
    )

    rows = SleepConsolidator(store, device="cpu")._memory_rows(4)

    assert [int(row["fact_id"]) for row in rows] == [
        pending,
        unbound,
        legacy,
        current,
    ]
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


def test_bounded_sleep_finalizes_bindings_and_records_checkpoint_lineage(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    pytest.importorskip("msgpack")
    store = MemoryStore(tmp_path / "memory.db")
    coordinator = MemoryCoordinator(store)
    first_memory = coordinator.ingest(
        "First cumulative sleep memory.",
        actor_type="user",
    )

    class FakeWindow:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def should_preempt(self):
            return False

        def __exit__(self, exc_type, exc, traceback):
            return None

    monkeypatch.setattr(
        "hippocampal_memory.sleep.ExclusiveSleepWindow", FakeWindow
    )
    consolidator = SleepConsolidator(
        store,
        state_root=tmp_path / "neural",
        circuit_config=tiny_config(),
        device="cpu",
        recording_stride=4,
    )
    first = consolidator.run_once(max_memories=1, nrem_cycles=1, rem_cycles=1)
    first_binding = store._conn.execute(
        "SELECT encoding_version FROM engram_bindings WHERE memory_id=?",
        (str(first_memory),),
    ).fetchone()
    assert first["status"] == "completed"
    assert first["parent_checkpoint_id"] is None
    assert first["frames"] == 11
    assert first_binding["encoding_version"] == "content-v3"

    second_memory = coordinator.ingest(
        "Second cumulative sleep memory.",
        actor_type="user",
    )
    second = consolidator.run_once(max_memories=1, nrem_cycles=1, rem_cycles=1)
    second_binding = store._conn.execute(
        "SELECT encoding_version FROM engram_bindings WHERE memory_id=?",
        (str(second_memory),),
    ).fetchone()
    assert second["status"] == "completed"
    assert second["parent_checkpoint_id"] == first["session_id"]
    assert second_binding["encoding_version"] == "content-v3"
    store.close()


def test_failed_recording_leaves_binding_pending_for_recovery(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    pytest.importorskip("msgpack")
    store = MemoryStore(tmp_path / "memory.db")
    coordinator = MemoryCoordinator(store)
    memory_id = coordinator.ingest(
        "Recover this encoding after a bounded recording failure.",
        actor_type="user",
    )

    class FakeWindow:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def should_preempt(self):
            return False

        def __exit__(self, exc_type, exc, traceback):
            return None

    class FullRecorder:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def append(self, payload):
            raise RuntimeError("recording reached configured size bound")

        def __exit__(self, exc_type, exc, traceback):
            return None

    real_recorder = sleep_module.FrameRecorder
    monkeypatch.setattr(sleep_module, "ExclusiveSleepWindow", FakeWindow)
    monkeypatch.setattr(sleep_module, "FrameRecorder", FullRecorder)
    consolidator = SleepConsolidator(
        store,
        state_root=tmp_path / "neural",
        circuit_config=tiny_config(),
        device="cpu",
        recording_stride=4,
    )

    with pytest.raises(RuntimeError, match="recording reached"):
        consolidator.run_once(max_memories=1, nrem_cycles=1, rem_cycles=1)
    failed_binding = store._conn.execute(
        "SELECT encoding_version FROM engram_bindings WHERE memory_id=?",
        (str(memory_id),),
    ).fetchone()
    assert failed_binding["encoding_version"].startswith("content-v3-pending:")
    assert (
        store._conn.execute("SELECT COUNT(*) FROM neural_checkpoints").fetchone()[0]
        == 0
    )

    monkeypatch.setattr(sleep_module, "FrameRecorder", real_recorder)
    recovered = consolidator.run_once(
        max_memories=1,
        nrem_cycles=1,
        rem_cycles=1,
    )
    recovered_binding = store._conn.execute(
        "SELECT encoding_version FROM engram_bindings WHERE memory_id=?",
        (str(memory_id),),
    ).fetchone()
    assert recovered["status"] == "completed"
    assert recovered["encoded"] == 1
    assert recovered_binding["encoding_version"] == "content-v3"
    store.close()
