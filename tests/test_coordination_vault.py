from pathlib import Path

import pytest

from hippocampal_memory.coordination import MemoryCoordinator, RevisionConflict
from hippocampal_memory.cognition import CognitiveMemorySystem
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
        context_id="context:refactor",
        event_start_at="2026-01-01T12:00:00+00:00",
        event_end_at="2026-01-01T12:05:00+00:00",
        autobiographical=True,
        self_relevance=0.8,
        perspective="field",
        recollection_mode="remember",
        vividness=0.7,
    )
    cognition = CognitiveMemorySystem(store, coordinator)
    cognition.segment_memories([memory_id], context_id="context:refactor")
    cognition.monitor_source(memory_id)
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
    assert 'context_id: "context:refactor"' in note.read_text()
    assert "recollection_mode: \"remember\"" in note.read_text()
    assert "## Temporal context" in note.read_text()
    assert "## Recollection" in note.read_text()
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


def test_vault_projects_bounded_explainable_neural_links(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    coordinator = MemoryCoordinator(store)
    atlas = coordinator.ingest(
        "Project Atlas uses port 8765 for the local observatory.",
        actor_type="user",
        context_id="context:atlas",
    )
    service = coordinator.ingest(
        "The Atlas observatory service listens on port 8765.",
        actor_type="user",
        context_id="context:atlas",
    )
    garden = coordinator.ingest(
        "Garden roses need water in the morning.",
        actor_type="user",
        context_id="context:garden",
    )
    coordinator.bind_engram(
        atlas,
        [1, 2, 3, 4],
        circuit_version="trisynaptic-v3-content-readout",
        encoding_version="content-v3",
        content_sha256="atlas",
        ca1_signature=[101, 102, 103, 104],
    )
    coordinator.bind_engram(
        service,
        [2, 3, 4, 5],
        circuit_version="trisynaptic-v3-content-readout",
        encoding_version="content-v3",
        content_sha256="service",
        ca1_signature=[102, 103, 104, 105],
    )
    coordinator.bind_engram(
        garden,
        [7, 8, 9],
        circuit_version="trisynaptic-v3-content-readout",
        encoding_version="content-v3",
        content_sha256="garden",
        ca1_signature=[103, 201, 202],
    )
    synchronizer = VaultSynchronizer(store, tmp_path / "vault", coordinator)

    plan = {item.memory_id: item for item in synchronizer.plan()}

    service_path = synchronizer.projector.note_path(store.get_fact(service))
    garden_path = synchronizer.projector.note_path(store.get_fact(garden))
    assert f"[[{service_path.removesuffix('.md')}|" in plan[str(atlas)].content
    assert f"[[{garden_path.removesuffix('.md')}|" not in plan[str(atlas)].content
    assert "neural overlap 0.750" in plan[str(atlas)].content
    assert "same context" in plan[str(atlas)].content
    assert "shared terms:" in plan[str(atlas)].content
    store.close()


def test_vault_neural_links_are_capped_per_note(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    coordinator = MemoryCoordinator(store)
    memory_ids = []
    for index in range(8):
        memory_id = coordinator.ingest(
            f"Shared project concept number {index}.",
            actor_type="user",
            context_id="context:shared",
        )
        coordinator.bind_engram(
            memory_id,
            [index + 1],
            circuit_version="trisynaptic-v3-content-readout",
            encoding_version="content-v3",
            content_sha256=str(index),
            ca1_signature=[101, 102, 103],
        )
        memory_ids.append(memory_id)
    synchronizer = VaultSynchronizer(store, tmp_path / "vault", coordinator)

    first = next(
        item
        for item in synchronizer.plan([memory_ids[0]])
        if item.memory_id == str(memory_ids[0])
    )

    assert first.content.count("[[") == 5
    store.close()
