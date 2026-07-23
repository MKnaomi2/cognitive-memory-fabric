from datetime import datetime, timedelta, timezone

import pytest

from hippocampal_memory.cognition import CognitiveMemorySystem
from hippocampal_memory.store import MemoryStore


def _episode(store, content, when, *, autobiographical=False):
    return store.add_fact(
        content,
        memory_kind="episode",
        provenance_type="agent",
        provenance_ref=f"session:{when}",
        valid_from=when,
        event_start_at=when,
        event_end_at=when,
        autobiographical=autobiographical,
        self_relevance=0.9 if autobiographical else 0.2,
        perspective="field" if autobiographical else "semantic",
        recollection_mode="remember" if autobiographical else "know",
        vividness=0.8 if autobiographical else 0.2,
    )


def test_event_segmentation_temporal_binding_and_order(tmp_path):
    store = MemoryStore(tmp_path / "memory.db", hrr_dim=64)
    cognition = CognitiveMemorySystem(store)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ids = [
        _episode(store, "Opened the deployment plan.", start.isoformat()),
        _episode(
            store,
            "Ran the deployment verification.",
            (start + timedelta(minutes=2)).isoformat(),
        ),
        _episode(
            store,
            "Discussed an unrelated travel itinerary.",
            (start + timedelta(hours=2)).isoformat(),
        ),
    ]

    events = cognition.segment_memories(ids, context_id="session:test", gap_seconds=900)

    assert len(events) == 2
    assert events[1]["boundary_reason"] == "temporal-gap"
    ordered = cognition.temporal_order(context_id="session:test")
    assert [row["fact_id"] for row in ordered] == ids
    bindings = store._conn.execute(
        "SELECT before_fact_id,after_fact_id FROM temporal_bindings"
    ).fetchall()
    assert [(row[0], row[1]) for row in bindings] == [(ids[0], ids[1])]
    store.close()


def test_extending_existing_context_rebuilds_order_idempotently(tmp_path):
    store = MemoryStore(tmp_path / "memory.db", hrr_dim=64)
    cognition = CognitiveMemorySystem(store)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = _episode(store, "Started the repair.", start.isoformat())
    second = _episode(
        store,
        "Verified the repair.",
        (start + timedelta(minutes=2)).isoformat(),
    )

    cognition.segment_memories([first], context_id="session:retry")
    cognition.segment_memories([second], context_id="session:retry")
    cognition.segment_memories([second], context_id="session:retry")

    ordered = cognition.temporal_order(context_id="session:retry")
    assert [row["fact_id"] for row in ordered] == [first, second]
    event_rows = store._conn.execute(
        """
        SELECT em.fact_id,em.sequence_index
        FROM event_memories em
        JOIN episodic_events ee ON ee.event_id=em.event_id
        WHERE ee.context_id=?
        ORDER BY ee.ordinal,em.sequence_index
        """,
        ("session:retry",),
    ).fetchall()
    assert [(row["fact_id"], row["sequence_index"]) for row in event_rows] == [
        (first, 0),
        (second, 1),
    ]
    store.close()


def test_recency_and_context_reinstatement(tmp_path):
    store = MemoryStore(tmp_path / "memory.db", hrr_dim=64)
    cognition = CognitiveMemorySystem(store)
    now = datetime(2026, 2, 1, tzinfo=timezone.utc)
    old = _episode(
        store, "Reviewed the older release.", (now - timedelta(days=7)).isoformat()
    )
    recent = _episode(
        store, "Reviewed the newest release.", (now - timedelta(hours=1)).isoformat()
    )
    cognition.segment_memories([old, recent], context_id="release-review")

    recalled = cognition.recall_recent(as_of=now, half_life_hours=24)
    assert recalled[0]["fact_id"] == recent
    assert recalled[0]["recency_score"] > recalled[1]["recency_score"]

    reinstated = cognition.reinstate_context(memory_id=old, cue="release")
    assert reinstated["context"]["context_id"] == "release-review"
    assert [row["fact_id"] for row in reinstated["memories"]] == [old, recent]
    store.close()


def test_source_monitoring_uses_provenance_and_corroboration(tmp_path):
    store = MemoryStore(tmp_path / "memory.db", hrr_dim=64)
    cognition = CognitiveMemorySystem(store)
    fact_id = store.add_fact(
        "The release passed acceptance.",
        provenance_type="user",
        provenance_ref="session:1",
        provenance={"message_ids": [1]},
    )
    store.record_evidence(
        fact_id,
        "confirm",
        provenance_type="system",
        provenance_ref="test-run:2",
    )
    store.record_evidence(
        fact_id,
        "confirm",
        provenance_type="web",
        provenance_ref="report:3",
    )

    assessment = cognition.monitor_source(fact_id)

    assert assessment["decision"] == "supported"
    assert assessment["independent_sources"] == 3
    assert assessment["source_memory_score"] >= 0.75
    store.close()


def test_reconsolidation_has_boundary_window_and_versions(tmp_path):
    store = MemoryStore(tmp_path / "memory.db", hrr_dim=64)
    cognition = CognitiveMemorySystem(store)
    when = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    memory_id = _episode(
        store, "I completed the migration.", when, autobiographical=True
    )
    cognition.segment_memories([memory_id], context_id="migration")

    short = cognition.reactivate(
        memory_id,
        cue="migration mismatch",
        prediction_error=0.8,
        retrieval_duration_seconds=1,
    )
    assert short["labile"] is False

    opened = cognition.reactivate(
        memory_id,
        cue="migration mismatch",
        prediction_error=0.8,
        retrieval_duration_seconds=300,
    )
    assert opened["labile"] is True
    result = cognition.reconsolidate(
        opened["window_id"],
        polarity="confirm",
        provenance_type="user",
        provenance_ref="session:correction",
        detail="The result was verified again.",
        contextual_updates={"vividness": 0.9, "recollection_mode": "remember"},
    )

    assert result["status"] == "reconsolidated"
    versions = store._conn.execute(
        "SELECT phase FROM memory_versions WHERE fact_id=? ORDER BY created_at",
        (memory_id,),
    ).fetchall()
    assert {row["phase"] for row in versions} == {"baseline", "restabilized"}
    assert store.get_fact(memory_id)["last_reconsolidated_at"] is not None
    store.close()


def test_autonoetic_profile_is_operational_not_consciousness_claim(tmp_path):
    store = MemoryStore(tmp_path / "memory.db", hrr_dim=64)
    cognition = CognitiveMemorySystem(store)
    _episode(
        store,
        "I noticed that decomposition improved my work.",
        datetime.now(timezone.utc).isoformat(),
        autobiographical=True,
    )

    profile = cognition.autonoetic_profile()

    assert profile["autobiographical_memories"] == 1
    assert profile["remember_mode"] == 1
    assert "not phenomenal consciousness" in profile["claim"]
    store.close()


def test_reconsolidation_rejects_unsupported_mutation(tmp_path):
    store = MemoryStore(tmp_path / "memory.db", hrr_dim=64)
    cognition = CognitiveMemorySystem(store)
    when = datetime.now(timezone.utc).isoformat()
    memory_id = _episode(store, "I ran a test.", when, autobiographical=True)
    cognition.segment_memories([memory_id], context_id="test")
    opened = cognition.reactivate(
        memory_id,
        cue="new result",
        prediction_error=0.9,
        retrieval_duration_seconds=300,
    )

    with pytest.raises(ValueError):
        cognition.reconsolidate(
            opened["window_id"],
            polarity="confirm",
            provenance_type="user",
            provenance_ref="session:1",
            detail="attempt",
            contextual_updates={"content": "overwrite"},
        )
    store.close()


def test_existing_memory_backfill_is_idempotent(tmp_path):
    store = MemoryStore(tmp_path / "memory.db", hrr_dim=64)
    first = store.add_fact(
        "First legacy observation.",
        provenance_type="user",
        provenance_ref="session:legacy",
        valid_from="2026-01-01T00:00:00+00:00",
    )
    second = store.add_fact(
        "Second legacy observation.",
        provenance_type="agent",
        provenance_ref="session:legacy",
        valid_from="2026-01-01T00:01:00+00:00",
    )
    cognition = CognitiveMemorySystem(store)

    initial = cognition.backfill_existing()
    repeated = cognition.backfill_existing()

    assert initial["source_assessments_created"] == 2
    assert repeated["source_assessments_created"] == 0
    assert store.get_fact(first)["context_id"] == "session:legacy"
    assert store.get_fact(second)["sequence_index"] == 1
    assert (
        store._conn.execute("SELECT COUNT(*) FROM temporal_bindings").fetchone()[0]
        == 1
    )
    store.close()
