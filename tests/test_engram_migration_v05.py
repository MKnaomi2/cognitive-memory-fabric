import json
from dataclasses import replace

import pytest

from hippocampal_memory.circuit import CircuitConfig
from hippocampal_memory.coordination import MemoryCoordinator
from hippocampal_memory.engram_migration import EngramMigrator
from hippocampal_memory.store import MemoryStore


def test_engram_migration_is_explicit_and_content_versioned(tmp_path):
    pytest.importorskip("torch")
    store = MemoryStore(tmp_path / "memory.db")
    memory_id = store.add_fact("Project Atlas uses port 9090.")
    coordinator = MemoryCoordinator(store)
    coordinator.bind_engram(
        memory_id,
        [1, 2, 3],
        circuit_version="trisynaptic-v2-time-cells",
    )
    tiny = replace(
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
    migrator = EngramMigrator(store, config=tiny)

    assert migrator.plan()[0]["memory_id"] == memory_id
    row = store._conn.execute(
        "SELECT * FROM engram_bindings WHERE memory_id=?", (str(memory_id),)
    ).fetchone()
    assert row["encoding_version"] == "memory-id-v2"

    result = migrator.apply()
    row = store._conn.execute(
        "SELECT * FROM engram_bindings WHERE memory_id=?", (str(memory_id),)
    ).fetchone()
    assert result["migrated"] == 1
    assert row["encoding_version"] == "content-v3"
    assert row["content_sha256"]
    assert isinstance(json.loads(row["ca1_signature_json"]), list)
    store.close()
