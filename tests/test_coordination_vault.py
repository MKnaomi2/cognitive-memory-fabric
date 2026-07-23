from pathlib import Path

import pytest

from hippocampal_memory.coordination import MemoryCoordinator, RevisionConflict
from hippocampal_memory.store import MemoryStore
from hippocampal_memory.vault import VaultSynchronizer


def test_events_are_idempotent_and_revision_guarded(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    coordinator = MemoryCoordinator(store)
    first = coordinator.append_event(
        "memory:1",
        "memory.observed",
        {"value": "x"},
        actor_type="user",
        event_id="fixed-event",
        expected_revision=0,
    )
    repeated = coordinator.append_event(
        "memory:1",
        "memory.observed",
        {"value": "x"},
        actor_type="user",
        event_id="fixed-event",
    )
    assert repeated == first
    assert first.revision == 1
    with pytest.raises(RevisionConflict):
        coordinator.append_event(
            "memory:1",
            "memory.updated",
            {"value": "y"},
            actor_type="agent",
            expected_revision=0,
        )
    store.close()


def test_vault_projection_preserves_human_notes(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    coordinator = MemoryCoordinator(store)
    memory_id = coordinator.ingest(
        "Hierarchical planning improves large refactors.",
        actor_type="reflection",
        actor_ref="replay:7",
        memory_kind="principle",
        confidence=0.82,
    )
    vault = tmp_path / "vault"
    synchronizer = VaultSynchronizer(store, vault, coordinator)
    first = synchronizer.plan([memory_id])
    assert len(first) == 1
    result = synchronizer.apply(first)
    assert result["count"] == 1

    note = vault / first[0].relative_path
    note.write_text(note.read_text() + "\n## Human notes\n\nMy annotation.\n")
    store.record_evidence(
        memory_id,
        "confirm",
        provenance_type="user",
        provenance_ref="session:later",
    )
    coordinator.append_event(
        f"memory:{memory_id}",
        "evidence.confirmed",
        {"source": "session:later"},
        actor_type="user",
    )
    second = synchronizer.plan([memory_id])
    assert len(second) == 1
    synchronizer.apply(second)
    assert "My annotation." in note.read_text()
    assert "sync_revision: 2" in note.read_text()
    store.close()


def test_vault_batch_limit_is_fail_closed(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    coordinator = MemoryCoordinator(store)
    for value in ("one", "two"):
        coordinator.ingest(value, actor_type="user")
    synchronizer = VaultSynchronizer(store, tmp_path / "vault", coordinator)
    with pytest.raises(ValueError):
        synchronizer.apply(synchronizer.plan(), max_mutations=1)
    assert not (tmp_path / "vault").exists()
    store.close()
