"""Deterministic policy and replay tests for the local hippocampus."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

from hippocampal_memory.replay import HippocampusEngine, ReplayConfig
from hippocampal_memory.store import MemoryStore


def _state_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, started_at REAL NOT NULL, ended_at REAL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT, tool_name TEXT, timestamp REAL NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        );
        INSERT INTO sessions VALUES ('s1', 1000, 2000);
        INSERT INTO messages VALUES
            (1, 's1', 'user', 'I prefer concise status reports.', NULL, 1001, 1),
            (2, 's1', 'assistant', 'Understood.', NULL, 1002, 1);
        """
    )
    conn.close()


class _FakeEngine(HippocampusEngine):
    def _gpu_busy(self):
        return False

    def _ollama_json(self, prompt, schema):
        if "Extract durable" in prompt:
            return {
                "memories": [
                    {
                        "content": "The user prefers concise status reports.",
                        "kind": "episode",
                        "category": "user_pref",
                        "confidence": 0.9,
                        "source_message_ids": [1],
                        "subject_key": "user",
                        "predicate_key": "status-report-style",
                        "expires_at": None,
                        "salience": 0.8,
                        "source_quality": 1.0,
                        "pinned": False,
                    }
                ]
            }
        return {"consolidations": [], "supersessions": []}


def test_backfill_is_grounded_and_checkpointed(tmp_path):
    state = tmp_path / "state.db"
    _state_db(state)
    store = MemoryStore(tmp_path / "memory.db", hrr_dim=64)
    engine = _FakeEngine(
        store=store, state_db=state, config=ReplayConfig(gpu_busy_threshold=101)
    )

    result = engine.run("backfill")

    assert result["status"] == "completed"
    assert result["memories_created"] == 1
    fact = store.list_facts()[0]
    assert fact["provenance_ref"] == "session:s1"
    assert fact["provenance"]["message_ids"] == [1]
    queue = store._conn.execute(
        "SELECT status, last_message_id FROM hippocampus_sessions WHERE session_id = 's1'"
    ).fetchone()
    assert queue["status"] == "done"
    assert queue["last_message_id"] == 2
    store.close()


def test_shadow_does_not_advance_checkpoint_or_write_memory(tmp_path):
    state = tmp_path / "state.db"
    _state_db(state)
    store = MemoryStore(tmp_path / "memory.db", hrr_dim=64)
    engine = _FakeEngine(store=store, state_db=state)

    result = engine.run("backfill", shadow=True)

    assert result["status"] == "completed"
    assert store.list_facts() == []
    queue = store._conn.execute(
        "SELECT status, last_message_id FROM hippocampus_sessions WHERE session_id = 's1'"
    ).fetchone()
    assert queue["status"] == "queued"
    assert queue["last_message_id"] == 0
    store.close()


def test_malformed_local_json_retries_once_with_larger_budget(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "memory.db", hrr_dim=64)
    engine = HippocampusEngine(store=store, state_db=tmp_path / "missing.db")
    calls = []

    def attempt(prompt, schema, *, num_predict):
        calls.append(num_predict)
        if len(calls) == 1:
            raise json.JSONDecodeError("malformed", "{", 1)
        return {"memories": []}

    monkeypatch.setattr(engine, "_ollama_json_attempt", attempt)

    assert engine._ollama_json("prompt", {}) == {"memories": []}
    assert calls == [2048, 4096]
    store.close()


def test_identity_requires_independent_sources_and_time_span(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ids = [
        store.add_fact(
            f"Episode {index}",
            provenance_ref=f"session:s{index}",
            memory_kind="episode",
            confidence=0.9,
            valid_from=(start + timedelta(days=index * 2)).isoformat(),
        )
        for index in range(5)
    ]

    assessment = store.assess_consolidation(ids, "identity")

    assert assessment["eligible"] is True
    assert assessment["independent_sources"] == 5
    assert assessment["span_days"] == 8
    store.close()


def test_supersession_requires_same_subject_predicate_and_quality(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    old = store.add_fact(
        "Service uses port 8080.",
        subject_key="service",
        predicate_key="port",
        source_quality=0.8,
    )
    new = store.add_fact(
        "Service uses port 9090.",
        subject_key="service",
        predicate_key="port",
        source_quality=0.9,
    )

    assert store.supersede_fact(old, new, "newer deployment record")
    fact = store.get_fact(old)
    assert fact["status"] == "archived"
    assert fact["superseded_by"] == new
    assert fact["valid_until"] is not None
    store.close()


def test_forgetting_archives_expired_but_not_pinned(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    expired = store.add_fact(
        "Temporary incident state.",
        expires_at="2000-01-01T00:00:00+00:00",
    )
    pinned = store.add_fact(
        "Never send email.",
        expires_at="2000-01-01T00:00:00+00:00",
        pinned=True,
    )

    result = store.run_forgetting_maintenance()

    assert expired in result["archived"]
    assert store.get_fact(expired)["status"] == "archived"
    assert store.get_fact(pinned)["status"] == "active"
    store.close()
