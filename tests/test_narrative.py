import json

import pytest

from hippocampal_memory.coordination import MemoryCoordinator
from hippocampal_memory.narrative import NarrativeEngine
from hippocampal_memory.narrative_evaluation import run_narrative_evaluation
from hippocampal_memory.store import MemoryStore
from hippocampal_memory.vault import VaultSynchronizer


def seeded_store(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    coordinator = MemoryCoordinator(store)
    ids = []
    contents = [
        "The project first selected a local SQLite memory store.",
        "The project then added provenance and confidence tracking.",
        "The project later enabled neural replay on the RTX GPU.",
        "The resulting system kept symbolic fallback for safety.",
    ]
    for index, content in enumerate(contents):
        memory_id = coordinator.ingest(
            content,
            actor_type="user",
            confidence=0.85,
            context_id="context:project",
            event_id="event:project",
            sequence_index=index,
            event_start_at=f"2026-07-2{index + 1}T12:00:00Z",
        )
        coordinator.bind_engram(
            memory_id,
            [index + 1],
            circuit_version="trisynaptic-v3-content-readout",
            encoding_version="content-v3",
            content_sha256=f"hash-{index}",
            ca1_signature=[10, 20, 30 + index],
        )
        ids.append(memory_id)
    return store, ids


def test_narrative_composes_bounded_cited_chronology(tmp_path):
    store, ids = seeded_store(tmp_path)
    result = NarrativeEngine(store).compose(
        "How did the project memory system develop?",
        max_memories=4,
    )

    assert result["structure"] == "chronological"
    assert result["memory_count"] == 4
    assert [row["memory_id"] for row in result["memories"]] == ids
    assert all(claim["source_ids"] for claim in result["claims"])
    assert "**Remembered:**" in result["story"]
    assert f"[m{ids[0]}]" in result["story"]
    store.close()


def test_neural_association_is_labeled_inference_not_causation(tmp_path):
    store, ids = seeded_store(tmp_path)
    other = MemoryCoordinator(store).ingest(
        "A separate deployment also used a guarded local GPU.",
        actor_type="user",
        confidence=0.8,
    )
    MemoryCoordinator(store).bind_engram(
        other,
        [99],
        circuit_version="trisynaptic-v3-content-readout",
        encoding_version="content-v3",
        content_sha256="other-hash",
        ca1_signature=[10, 20, 99],
    )

    result = NarrativeEngine(store).compose("guarded local GPU deployment")
    associations = [
        claim
        for claim in result["claims"]
        if claim["relation"] == "neural_association"
    ]
    assert associations
    assert all(claim["kind"] == "inference" for claim in associations)
    assert all("does not establish causation" in claim["text"] for claim in associations)
    assert other in {row["memory_id"] for row in result["memories"]}
    assert ids
    store.close()


def test_open_conflict_is_preserved_in_story(tmp_path):
    store, ids = seeded_store(tmp_path)
    conflict = MemoryCoordinator(store).ingest(
        "The project did not retain symbolic fallback.",
        actor_type="user",
        confidence=0.8,
        context_id="context:project",
        event_id="event:project",
        sequence_index=4,
    )
    store.record_conflict(ids[-1], conflict, "opposing fallback claims")

    result = NarrativeEngine(store).compose("symbolic fallback safety")

    claims = [claim for claim in result["claims"] if claim["relation"] == "conflicts"]
    assert claims
    assert claims[0]["kind"] == "remembered"
    assert "both versions" in claims[0]["text"]
    store.close()


def test_sleep_narratives_draft_then_promote_and_become_stale(tmp_path):
    store, ids = seeded_store(tmp_path)
    engine = NarrativeEngine(store)

    first = engine.consolidate_drafts("sleep-one")
    assert first["created"] == 1
    assert engine.list_threads(status="draft")[0]["support_passes"] == 1

    second = engine.consolidate_drafts("sleep-two")
    active = engine.list_threads(status="active")
    assert second["promoted"] == 1
    assert active[0]["support_passes"] == 2

    database = store.db_path
    store.close()
    store = MemoryStore(database)
    engine = NarrativeEngine(store)
    assert engine.list_threads(status="active")[0]["thread_id"] == active[0]["thread_id"]

    store.archive_fact(ids[0], "source became obsolete")
    assert engine.refresh_stale() == 1
    assert engine.list_threads(status="stale")[0]["thread_id"] == active[0]["thread_id"]
    store.close()


def test_explicit_helpful_feedback_can_promote_supported_draft(tmp_path):
    store, _ = seeded_store(tmp_path)
    engine = NarrativeEngine(store)
    engine.consolidate_drafts("sleep-one")
    draft = engine.list_threads(status="draft")[0]

    result = engine.feedback(rating="helpful", thread_id=draft["thread_id"])

    assert result["promoted"] is True
    assert engine.list_threads(status="active")[0]["helpful_count"] == 1
    store.close()


def test_vault_projects_promoted_narratives_not_drafts(tmp_path):
    store, ids = seeded_store(tmp_path)
    engine = NarrativeEngine(store)
    vault = tmp_path / "vault"
    engine.consolidate_drafts("sleep-one")
    draft_plan = VaultSynchronizer(store, vault).plan()
    assert not any("Narratives/Active" in item.relative_path for item in draft_plan)

    engine.consolidate_drafts("sleep-two")
    active_plan = VaultSynchronizer(store, vault).plan()
    narrative = next(
        item for item in active_plan if "Narratives/Active" in item.relative_path
    )
    assert narrative.memory_id.startswith("narrative:")
    assert "## Supporting memories" in narrative.content
    VaultSynchronizer(store, vault).apply(
        active_plan, max_mutations=len(active_plan)
    )

    store.archive_fact(ids[0], "source became obsolete")
    assert engine.refresh_stale() == 1
    stale_plan = VaultSynchronizer(store, vault).plan()
    stale = next(item for item in stale_plan if item.memory_id == narrative.memory_id)
    assert stale.relative_path == narrative.relative_path
    assert "status: stale" in stale.content
    assert "must be revalidated" in stale.content
    store.close()


def test_private_narrative_evaluation_is_hashed_and_split(tmp_path):
    store, _ = seeded_store(tmp_path)
    database = store.db_path
    store.close()

    result = run_narrative_evaluation(
        database,
        tmp_path / "evaluation",
        cases=30,
        split="development",
    )
    manifest = json.loads(
        (tmp_path / "evaluation" / "manifest.json").read_text(encoding="utf-8")
    )

    assert result["status"] == "completed"
    assert 0 < result["cases"] < 30
    assert result["trials"] == result["cases"] * 2
    assert set(result["condition_metrics"]) == {
        "evidence-only",
        "evidence-plus-neural",
    }
    assert result["citation_precision_mean"] == pytest.approx(1.0)
    assert manifest["artifacts"]["trials.jsonl"]
    assert result["privacy"].startswith("raw cases are local-only")
