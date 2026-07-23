"""Lifecycle tests for provenance-aware holographic memory."""

from hippocampal_memory.store import MemoryStore


def test_every_memory_has_provenance_and_confidence(tmp_path):
    store = MemoryStore(tmp_path / "memory.db", default_trust=0.4)
    fact_id = store.add_fact(
        "The deployment uses blue-green releases.",
        provenance_type="web",
        provenance_ref="https://example.test/runbook",
        provenance={"title": "Deployment runbook"},
        memory_kind="episode",
    )

    fact = store.get_fact(fact_id)
    assert fact["provenance_type"] == "web"
    assert fact["provenance_ref"] == "https://example.test/runbook"
    assert fact["provenance"]["title"] == "Deployment runbook"
    assert fact["memory_kind"] == "episode"
    assert fact["confidence"] == 0.4
    store.close()


def test_confirmation_and_contradiction_change_confidence(tmp_path):
    store = MemoryStore(tmp_path / "memory.db", default_trust=0.5)
    fact_id = store.add_fact("Large refactors benefit from hierarchical planning.")

    confirmed = store.record_evidence(
        fact_id, "confirm", provenance_type="agent", provenance_ref="task-2"
    )
    contradicted = store.record_evidence(
        fact_id, "contradict", provenance_type="user", provenance_ref="session-3"
    )

    assert confirmed["confidence"] > 0.5
    assert contradicted["confidence"] < confirmed["confidence"]
    fact = store.get_fact(fact_id)
    assert fact["confirmation_count"] == 1
    assert fact["contradiction_count"] == 1
    store.close()


def test_remove_archives_and_restore_recovers(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    fact_id = store.add_fact("Archive me.", provenance_type="user")

    assert store.remove_fact(fact_id)
    assert store.get_fact(fact_id)["status"] == "archived"
    assert store.list_facts() == []
    assert store.list_facts(include_archived=True)[0]["fact_id"] == fact_id

    assert store.restore_fact(fact_id)
    assert store.get_fact(fact_id)["status"] == "active"
    store.close()


def test_consolidation_preserves_lineage_and_identity(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    first = store.add_fact("I decomposed refactor A first.", memory_kind="episode")
    second = store.add_fact("I decomposed migration B first.", memory_kind="episode")

    identity = store.consolidate(
        "I consistently perform better when I decompose large tasks first.",
        [first, second],
        memory_kind="identity",
    )

    assert store.get_fact(identity)["memory_kind"] == "identity"
    assert store.get_fact(first)["status"] == "archived"
    sources = store._conn.execute(
        "SELECT source_fact_id FROM fact_derivations WHERE derived_fact_id = ?",
        (identity,),
    ).fetchall()
    assert {row["source_fact_id"] for row in sources} == {first, second}
    store.close()


def test_conflict_preserves_both_memories(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    first = store.add_fact("The service uses port 8080.")
    second = store.add_fact("The service uses port 9090.")

    conflict_id = store.record_conflict(first, second, "Same service, different port")

    conflicts = store.list_conflicts()
    assert conflicts[0]["conflict_id"] == conflict_id
    assert store.get_fact(first)["status"] == "conflicted"
    assert store.get_fact(second)["status"] == "conflicted"
    assert store.get_fact(first)["contradiction_count"] == 1
    assert store.get_fact(first)["confidence"] < 0.5
    store.close()


def test_duplicate_observation_is_confirmation(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    first = store.add_fact(
        "User prefers concise status updates.", provenance_type="user"
    )
    second = store.add_fact(
        "User prefers concise status updates.", provenance_type="user"
    )

    assert first == second
    assert store.get_fact(first)["confirmation_count"] == 1
    assert store.get_fact(first)["confidence"] > 0.5
    store.close()
