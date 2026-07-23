"""Operational temporal, episodic, source, and reconsolidation processes.

These mechanisms model memory functions; they do not claim phenomenal
consciousness or biological equivalence.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from .coordination import MemoryCoordinator
from .store import MemoryStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS temporal_contexts (
    context_id TEXT PRIMARY KEY,
    context_type TEXT NOT NULL DEFAULT 'episode',
    started_at TEXT NOT NULL,
    ended_at TEXT,
    location TEXT NOT NULL DEFAULT '',
    participants_json TEXT NOT NULL DEFAULT '[]',
    cues_json TEXT NOT NULL DEFAULT '{}',
    affect_json TEXT NOT NULL DEFAULT '{}',
    summary TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodic_events (
    event_id TEXT PRIMARY KEY,
    context_id TEXT NOT NULL REFERENCES temporal_contexts(context_id),
    ordinal INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    boundary_reason TEXT NOT NULL,
    narrative_thread TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(context_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_episodic_events_context
    ON episodic_events(context_id, ordinal);
CREATE TABLE IF NOT EXISTS event_memories (
    event_id TEXT NOT NULL REFERENCES episodic_events(event_id),
    fact_id INTEGER NOT NULL REFERENCES facts(fact_id),
    sequence_index INTEGER NOT NULL,
    offset_seconds REAL NOT NULL DEFAULT 0,
    role TEXT NOT NULL DEFAULT 'content',
    PRIMARY KEY(event_id, fact_id),
    UNIQUE(event_id, sequence_index)
);
CREATE TABLE IF NOT EXISTS temporal_bindings (
    binding_id TEXT PRIMARY KEY,
    context_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    before_fact_id INTEGER NOT NULL REFERENCES facts(fact_id),
    after_fact_id INTEGER NOT NULL REFERENCES facts(fact_id),
    relation TEXT NOT NULL,
    delta_seconds REAL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(before_fact_id, after_fact_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_temporal_bindings_context
    ON temporal_bindings(context_id, event_id);
CREATE TABLE IF NOT EXISTS source_monitoring_assessments (
    assessment_id TEXT PRIMARY KEY,
    fact_id INTEGER NOT NULL REFERENCES facts(fact_id),
    provenance_type TEXT NOT NULL,
    provenance_ref TEXT NOT NULL DEFAULT '',
    prior_score REAL NOT NULL,
    completeness_score REAL NOT NULL,
    corroboration_score REAL NOT NULL,
    consistency_score REAL NOT NULL,
    temporal_score REAL NOT NULL,
    source_memory_score REAL NOT NULL,
    decision TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    assessed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_monitoring_fact
    ON source_monitoring_assessments(fact_id, assessed_at);
CREATE TABLE IF NOT EXISTS reconsolidation_windows (
    window_id TEXT PRIMARY KEY,
    fact_id INTEGER NOT NULL REFERENCES facts(fact_id),
    opened_at TEXT NOT NULL,
    closes_at TEXT NOT NULL,
    trigger TEXT NOT NULL,
    prediction_error REAL NOT NULL,
    retrieval_duration_seconds REAL NOT NULL,
    baseline_sha256 TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'labile',
    closed_at TEXT,
    detail_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_reconsolidation_status
    ON reconsolidation_windows(status, closes_at);
CREATE TABLE IF NOT EXISTS memory_versions (
    version_id TEXT PRIMARY KEY,
    fact_id INTEGER NOT NULL REFERENCES facts(fact_id),
    window_id TEXT,
    phase TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    snapshot_sha256 TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_versions_fact
    ON memory_versions(fact_id, created_at);
CREATE TABLE IF NOT EXISTS context_reinstatements (
    reinstatement_id TEXT PRIMARY KEY,
    cue TEXT NOT NULL,
    context_id TEXT NOT NULL,
    score REAL NOT NULL,
    memory_ids_json TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);
"""

_SOURCE_PRIORS = {
    "user": 0.95,
    "sensor": 0.85,
    "system": 0.80,
    "web": 0.70,
    "agent": 0.62,
    "reflection": 0.58,
    "imported": 0.50,
}
_PERSPECTIVES = {"field", "observer", "semantic", "unknown"}
_RECOLLECTION_MODES = {"remember", "know", "inferred", "unknown"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif value:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    else:
        result = datetime.now(timezone.utc)
    return result.replace(tzinfo=result.tzinfo or timezone.utc)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _tokens(value: str) -> set[str]:
    return {
        token.strip(".,;:!?\"'()[]{}").casefold()
        for token in value.split()
        if token.strip(".,;:!?\"'()[]{}")
    }


def _overlap(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a and b else 0.0


class CognitiveMemorySystem:
    """Deterministic cognitive-process layer over the lifecycle store."""

    def __init__(
        self, store: MemoryStore, coordinator: MemoryCoordinator | None = None
    ) -> None:
        self.store = store
        self.coordinator = coordinator or MemoryCoordinator(store)
        with store._lock:
            store._conn.executescript(_SCHEMA)

    def create_context(
        self,
        *,
        context_id: str | None = None,
        started_at: str | datetime | None = None,
        ended_at: str | datetime | None = None,
        context_type: str = "episode",
        location: str = "",
        participants: Iterable[str] = (),
        cues: dict[str, Any] | None = None,
        affect: dict[str, Any] | None = None,
        summary: str = "",
    ) -> str:
        """Persist the who/where/when/cue envelope around an experience."""
        context_id = context_id or str(uuid.uuid4())
        start = _parse(started_at).isoformat()
        end = _parse(ended_at).isoformat() if ended_at else None
        if end and _parse(end) < _parse(start):
            raise ValueError("context end precedes context start")
        with self.store._lock:
            self.store._conn.execute(
                """
                INSERT INTO temporal_contexts (
                    context_id,context_type,started_at,ended_at,location,
                    participants_json,cues_json,affect_json,summary,created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(context_id) DO UPDATE SET
                    ended_at=COALESCE(excluded.ended_at,temporal_contexts.ended_at),
                    location=CASE WHEN excluded.location!='' THEN excluded.location
                                  ELSE temporal_contexts.location END,
                    participants_json=excluded.participants_json,
                    cues_json=excluded.cues_json,
                    affect_json=excluded.affect_json,
                    summary=CASE WHEN excluded.summary!='' THEN excluded.summary
                                 ELSE temporal_contexts.summary END
                """,
                (
                    context_id,
                    context_type,
                    start,
                    end,
                    location,
                    _json(sorted(set(participants))),
                    _json(cues or {}),
                    _json(affect or {}),
                    summary[:2000],
                    _now(),
                ),
            )
        return context_id

    def segment_memories(
        self,
        memory_ids: Iterable[int],
        *,
        context_id: str | None = None,
        gap_seconds: float = 1800,
        topic_threshold: float = 0.08,
    ) -> list[dict[str, Any]]:
        """Segment ordered observations at temporal gaps or strong topic shifts."""
        ids = list(dict.fromkeys(int(value) for value in memory_ids))
        if not ids:
            return []
        if context_id:
            with self.store._lock:
                existing_ids = self.store._conn.execute(
                    "SELECT fact_id FROM facts WHERE context_id=?",
                    (context_id,),
                ).fetchall()
            ids = list(
                dict.fromkeys(
                    [*ids, *(int(row["fact_id"]) for row in existing_ids)]
                )
            )
        placeholders = ",".join("?" for _ in ids)
        with self.store._lock:
            rows = self.store._conn.execute(
                f"""
                SELECT fact_id,content,provenance_ref,
                       COALESCE(event_start_at,valid_from,created_at) occurred_at,
                       COALESCE(event_end_at,event_start_at,valid_from,created_at) ended_at
                FROM facts WHERE fact_id IN ({placeholders})
                ORDER BY occurred_at,fact_id
                """,
                ids,
            ).fetchall()
        if len(rows) != len(ids):
            raise KeyError("one or more memories do not exist")
        context_id = self.create_context(
            context_id=context_id,
            started_at=rows[0]["occurred_at"],
            ended_at=rows[-1]["ended_at"],
            context_type="autobiographical-session",
            cues={"source_refs": sorted({row["provenance_ref"] for row in rows})},
            summary=" ".join(str(row["content"]) for row in rows)[:2000],
        )
        groups: list[tuple[str, list[Any]]] = []
        current: list[Any] = []
        reason = "context-start"
        for row in rows:
            if current:
                gap = (
                    _parse(row["occurred_at"]) - _parse(current[-1]["ended_at"])
                ).total_seconds()
                shift = _overlap(str(current[-1]["content"]), str(row["content"]))
                if gap > gap_seconds or (gap > 120 and shift < topic_threshold):
                    groups.append((reason, current))
                    current = []
                    reason = "temporal-gap" if gap > gap_seconds else "topic-shift"
            current.append(row)
        groups.append((reason, current))

        with self.store._lock:
            old_events = self.store._conn.execute(
                "SELECT event_id FROM episodic_events WHERE context_id=?",
                (context_id,),
            ).fetchall()
            self.store._conn.execute(
                "DELETE FROM temporal_bindings WHERE context_id=?", (context_id,)
            )
            for old in old_events:
                self.store._conn.execute(
                    "DELETE FROM event_memories WHERE event_id=?", (old["event_id"],)
                )
            self.store._conn.execute(
                "DELETE FROM episodic_events WHERE context_id=?", (context_id,)
            )

        events = []
        for ordinal, (boundary, members) in enumerate(groups):
            event_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{context_id}:{ordinal}"))
            with self.store._lock:
                self.store._conn.execute(
                    """
                    INSERT INTO episodic_events (
                        event_id,context_id,ordinal,started_at,ended_at,
                        boundary_reason,narrative_thread,metadata_json
                    ) VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(event_id) DO UPDATE SET
                        ended_at=excluded.ended_at,
                        boundary_reason=excluded.boundary_reason
                    """,
                    (
                        event_id,
                        context_id,
                        ordinal,
                        members[0]["occurred_at"],
                        members[-1]["ended_at"],
                        boundary,
                        context_id,
                        _json({"segmentation": "gap+lexical-v1"}),
                    ),
                )
                start = _parse(members[0]["occurred_at"])
                for sequence, member in enumerate(members):
                    offset = (_parse(member["occurred_at"]) - start).total_seconds()
                    self.store._conn.execute(
                        """
                        INSERT INTO event_memories(
                            event_id,fact_id,sequence_index,offset_seconds
                        ) VALUES(?,?,?,?)
                        ON CONFLICT(event_id,fact_id) DO UPDATE SET
                            sequence_index=excluded.sequence_index,
                            offset_seconds=excluded.offset_seconds
                        """,
                        (event_id, member["fact_id"], sequence, offset),
                    )
                    self.store._conn.execute(
                        """
                        UPDATE facts SET context_id=?,event_id=?,sequence_index=?,
                            event_start_at=COALESCE(event_start_at,?),
                            event_end_at=COALESCE(event_end_at,?),
                            updated_at=CURRENT_TIMESTAMP
                        WHERE fact_id=?
                        """,
                        (
                            context_id,
                            event_id,
                            sequence,
                            member["occurred_at"],
                            member["ended_at"],
                            member["fact_id"],
                        ),
                    )
                self._bind_adjacent(context_id, event_id, members)
            events.append(
                {
                    "event_id": event_id,
                    "context_id": context_id,
                    "ordinal": ordinal,
                    "boundary_reason": boundary,
                    "memory_ids": [int(row["fact_id"]) for row in members],
                }
            )
        return events

    def _bind_adjacent(self, context_id: str, event_id: str, rows: list[Any]) -> None:
        for before, after in zip(rows, rows[1:]):
            delta = (
                _parse(after["occurred_at"]) - _parse(before["ended_at"])
            ).total_seconds()
            confidence = max(0.05, min(1.0, math.exp(-abs(delta) / 86400)))
            self.store._conn.execute(
                """
                INSERT OR IGNORE INTO temporal_bindings(
                    binding_id,context_id,event_id,before_fact_id,after_fact_id,
                    relation,delta_seconds,confidence,created_at
                ) VALUES(?,?,?,?,?,'before',?,?,?)
                """,
                (
                    str(uuid.uuid4()),
                    context_id,
                    event_id,
                    before["fact_id"],
                    after["fact_id"],
                    delta,
                    confidence,
                    _now(),
                ),
            )

    def temporal_order(
        self, *, context_id: str | None = None, event_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return an explicit temporal-order memory, never insertion order."""
        if not context_id and not event_id:
            raise ValueError("context_id or event_id is required")
        clause, value = (
            ("e.context_id", context_id) if context_id else ("em.event_id", event_id)
        )
        with self.store._lock:
            rows = self.store._conn.execute(
                f"""
                SELECT f.*,e.ordinal event_ordinal,em.sequence_index,
                       em.offset_seconds,e.boundary_reason
                FROM event_memories em
                JOIN episodic_events e ON e.event_id=em.event_id
                JOIN facts f ON f.fact_id=em.fact_id
                WHERE {clause}=?
                ORDER BY e.ordinal,em.sequence_index
                """,
                (value,),
            ).fetchall()
        return [self.store._row_to_dict(row) for row in rows]

    def recall_recent(
        self,
        *,
        limit: int = 20,
        half_life_hours: float = 168,
        context_id: str | None = None,
        as_of: str | datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Recency memory with an explicit, inspectable decay score."""
        now = _parse(as_of)
        params: list[Any] = []
        clause = "AND context_id=?" if context_id else ""
        if context_id:
            params.append(context_id)
        with self.store._lock:
            rows = self.store._conn.execute(
                f"""
                SELECT * FROM facts WHERE status!='archived' {clause}
                ORDER BY COALESCE(event_end_at,valid_from,updated_at) DESC
                LIMIT 500
                """,
                params,
            ).fetchall()
        scored = []
        half_life = max(0.01, float(half_life_hours))
        for row in rows:
            memory = self.store._row_to_dict(row)
            stamp = _parse(
                memory.get("event_end_at")
                or memory.get("valid_from")
                or memory.get("updated_at")
            )
            age_hours = max(0.0, (now - stamp).total_seconds() / 3600)
            memory["age_hours"] = round(age_hours, 4)
            memory["recency_score"] = round(
                math.pow(0.5, age_hours / half_life), 8
            )
            scored.append(memory)
        scored.sort(
            key=lambda item: (item["recency_score"], item["confidence"]), reverse=True
        )
        return scored[: max(1, min(100, int(limit)))]

    def monitor_source(self, fact_id: int) -> dict[str, Any]:
        """Evaluate source memory without erasing the original attribution."""
        with self.store._lock:
            fact = self.store._conn.execute(
                "SELECT * FROM facts WHERE fact_id=?", (int(fact_id),)
            ).fetchone()
            if not fact:
                raise KeyError(f"fact_id {fact_id} not found")
            evidence = self.store._conn.execute(
                "SELECT * FROM fact_evidence WHERE fact_id=?", (int(fact_id),)
            ).fetchall()
        source_type = str(fact["provenance_type"])
        prior = _SOURCE_PRIORS.get(source_type, 0.45)
        completeness = (
            float(bool(fact["provenance_ref"])) * 0.6
            + float(bool(fact["provenance_json"] not in {"", "{}"})) * 0.4
        )
        positive_refs = {
            str(row["source_ref"])
            for row in evidence
            if row["polarity"] in {"source", "confirm"} and row["source_ref"]
        }
        corroboration = min(1.0, len(positive_refs) / 3)
        positive = sum(
            float(row["weight"])
            for row in evidence
            if row["polarity"] in {"source", "confirm"}
        )
        negative = sum(
            float(row["weight"])
            for row in evidence
            if row["polarity"] == "contradict"
        )
        consistency = positive / max(1.0, positive + negative)
        temporal = 1.0
        if fact["valid_from"] and fact["valid_until"]:
            temporal = float(_parse(fact["valid_from"]) <= _parse(fact["valid_until"]))
        score = (
            0.35 * prior
            + 0.15 * completeness
            + 0.25 * corroboration
            + 0.15 * consistency
            + 0.10 * temporal
        )
        decision = "supported" if score >= 0.75 else "uncertain" if score >= 0.5 else "weak"
        result = {
            "fact_id": int(fact_id),
            "prior_score": round(prior, 4),
            "completeness_score": round(completeness, 4),
            "corroboration_score": round(corroboration, 4),
            "consistency_score": round(consistency, 4),
            "temporal_score": round(temporal, 4),
            "source_memory_score": round(score, 4),
            "decision": decision,
            "independent_sources": len(positive_refs),
        }
        with self.store._lock:
            self.store._conn.execute(
                """
                INSERT INTO source_monitoring_assessments VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    str(uuid.uuid4()),
                    int(fact_id),
                    source_type,
                    str(fact["provenance_ref"] or ""),
                    prior,
                    completeness,
                    corroboration,
                    consistency,
                    temporal,
                    score,
                    decision,
                    _json(result),
                    _now(),
                ),
            )
        return result

    def reinstate_context(
        self,
        *,
        cue: str = "",
        memory_id: int | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Reconstruct a temporal context and its ordered episodic contents."""
        context_scores: dict[str, float] = {}
        with self.store._lock:
            if memory_id is not None:
                row = self.store._conn.execute(
                    "SELECT context_id FROM facts WHERE fact_id=?", (memory_id,)
                ).fetchone()
                if not row:
                    raise KeyError(f"fact_id {memory_id} not found")
                if row["context_id"]:
                    context_scores[str(row["context_id"])] = 1.0
            contexts = self.store._conn.execute(
                "SELECT * FROM temporal_contexts"
            ).fetchall()
            for context in contexts:
                if context["context_id"] in context_scores:
                    continue
                material = (
                    str(context["summary"])
                    + " "
                    + str(context["cues_json"])
                    + " "
                    + str(context["location"])
                )
                context_scores[str(context["context_id"])] = _overlap(cue, material)
        if not context_scores:
            return {"context": None, "memories": [], "score": 0.0}
        context_id, score = max(context_scores.items(), key=lambda item: item[1])
        memories = self.temporal_order(context_id=context_id)[:limit]
        ids = [int(item["fact_id"]) for item in memories]
        with self.store._lock:
            self.store._conn.execute(
                """
                INSERT INTO context_reinstatements VALUES(?,?,?,?,?,?)
                """,
                (
                    str(uuid.uuid4()),
                    cue[:2000],
                    context_id,
                    score,
                    _json(ids),
                    _now(),
                ),
            )
            if ids:
                placeholders = ",".join("?" for _ in ids)
                self.store._conn.execute(
                    f"UPDATE facts SET last_reinstated_at=CURRENT_TIMESTAMP "
                    f"WHERE fact_id IN ({placeholders})",
                    ids,
                )
            context = self.store._conn.execute(
                "SELECT * FROM temporal_contexts WHERE context_id=?", (context_id,)
            ).fetchone()
        return {
            "context": dict(context) if context else None,
            "memories": memories,
            "score": round(score, 6),
        }

    def reactivate(
        self,
        memory_id: int,
        *,
        cue: str,
        prediction_error: float = 0.0,
        retrieval_duration_seconds: float = 0.0,
        window_hours: float = 6.0,
    ) -> dict[str, Any]:
        """Retrieve a memory and open lability only when boundary conditions pass."""
        fact = self.store.get_fact(memory_id)
        if not fact:
            raise KeyError(f"fact_id {memory_id} not found")
        reinstated = self.reinstate_context(cue=cue, memory_id=memory_id)
        age_days = max(
            0.0,
            (
                datetime.now(timezone.utc)
                - _parse(fact.get("event_start_at") or fact.get("created_at"))
            ).total_seconds()
            / 86400,
        )
        required_duration = 20 + 80 * float(fact["confidence"]) + min(180, age_days / 2)
        error = max(0.0, min(1.0, float(prediction_error)))
        labile = (
            fact["status"] != "archived"
            and error >= 0.20
            and float(retrieval_duration_seconds) >= required_duration
        )
        result = {
            "memory_id": memory_id,
            "reinstatement": reinstated,
            "labile": labile,
            "required_duration_seconds": round(required_duration, 2),
            "boundary_reason": (
                "prediction error and retrieval duration satisfied"
                if labile
                else "retrieval did not satisfy reconsolidation boundary conditions"
            ),
        }
        if not labile:
            return result
        snapshot = self._snapshot(memory_id)
        window_id = str(uuid.uuid4())
        opened = datetime.now(timezone.utc)
        closes = opened + timedelta(hours=max(1.0, min(12.0, window_hours)))
        digest = hashlib.sha256(_json(snapshot).encode()).hexdigest()
        with self.store._lock:
            self.store._conn.execute(
                """
                INSERT INTO reconsolidation_windows
                VALUES(?,?,?,?,?,?,?,?,?,NULL,?)
                """,
                (
                    window_id,
                    memory_id,
                    opened.isoformat(),
                    closes.isoformat(),
                    cue[:1000],
                    error,
                    float(retrieval_duration_seconds),
                    digest,
                    "labile",
                    _json({"required_duration_seconds": required_duration}),
                ),
            )
            self._version(memory_id, window_id, "baseline", snapshot, "reactivation")
        self.coordinator.append_event(
            f"memory:{memory_id}",
            "memory.reactivated",
            {
                "window_id": window_id,
                "prediction_error": error,
                "closes_at": closes.isoformat(),
            },
            actor_type="system",
            actor_ref="reconsolidation-policy",
        )
        result["window_id"] = window_id
        result["closes_at"] = closes.isoformat()
        return result

    def reconsolidate(
        self,
        window_id: str,
        *,
        polarity: str,
        provenance_type: str,
        provenance_ref: str,
        detail: str,
        weight: float = 1.0,
        contextual_updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Restabilize a labile memory while retaining before/after versions."""
        with self.store._lock:
            window = self.store._conn.execute(
                "SELECT * FROM reconsolidation_windows WHERE window_id=?",
                (window_id,),
            ).fetchone()
        if not window:
            raise KeyError("reconsolidation window not found")
        if window["status"] != "labile":
            raise ValueError("reconsolidation window is not labile")
        if _parse(window["closes_at"]) < datetime.now(timezone.utc):
            with self.store._lock:
                self.store._conn.execute(
                    "UPDATE reconsolidation_windows SET status='expired',"
                    "closed_at=? WHERE window_id=?",
                    (_now(), window_id),
                )
            raise ValueError("reconsolidation window expired")
        fact_id = int(window["fact_id"])
        before = self._snapshot(fact_id)
        evidence = self.store.record_evidence(
            fact_id,
            polarity,
            provenance_type=provenance_type,
            provenance_ref=provenance_ref,
            detail=detail,
            weight=weight,
        )
        updates = contextual_updates or {}
        allowed = {
            "event_start_at",
            "event_end_at",
            "temporal_uncertainty_seconds",
            "perspective",
            "recollection_mode",
            "vividness",
            "self_relevance",
        }
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"unsupported reconsolidation fields: {sorted(unknown)}")
        if updates.get("perspective") not in {None, *_PERSPECTIVES}:
            raise ValueError("invalid perspective")
        if updates.get("recollection_mode") not in {None, *_RECOLLECTION_MODES}:
            raise ValueError("invalid recollection_mode")
        with self.store._lock:
            for key, value in updates.items():
                if key in {"vividness", "self_relevance"}:
                    value = max(0.0, min(1.0, float(value)))
                if key == "temporal_uncertainty_seconds":
                    value = max(0.0, float(value))
                self.store._conn.execute(
                    f"UPDATE facts SET {key}=?,updated_at=CURRENT_TIMESTAMP "
                    "WHERE fact_id=?",
                    (value, fact_id),
                )
            self.store._conn.execute(
                "UPDATE facts SET last_reconsolidated_at=CURRENT_TIMESTAMP "
                "WHERE fact_id=?",
                (fact_id,),
            )
            after = self._snapshot(fact_id)
            self._version(
                fact_id, window_id, "restabilized", after, "evidence-integrated"
            )
            self.store._conn.execute(
                """
                UPDATE reconsolidation_windows SET status='reconsolidated',
                    closed_at=?,detail_json=? WHERE window_id=?
                """,
                (
                    _now(),
                    _json({"evidence": evidence, "updates": updates}),
                    window_id,
                ),
            )
        self.coordinator.append_event(
            f"memory:{fact_id}",
            "memory.reconsolidated",
            {
                "window_id": window_id,
                "before_sha256": hashlib.sha256(_json(before).encode()).hexdigest(),
                "after_sha256": hashlib.sha256(_json(after).encode()).hexdigest(),
            },
            actor_type="system",
            actor_ref="reconsolidation-policy",
        )
        return {
            "memory_id": fact_id,
            "window_id": window_id,
            "status": "reconsolidated",
            "evidence": evidence,
        }

    def autobiographical_timeline(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return self-relevant episodes in their experienced temporal order."""
        with self.store._lock:
            rows = self.store._conn.execute(
                """
                SELECT f.*,e.ordinal event_ordinal,em.sequence_index
                FROM facts f
                LEFT JOIN event_memories em ON em.fact_id=f.fact_id
                LEFT JOIN episodic_events e ON e.event_id=em.event_id
                WHERE f.status!='archived' AND
                      (f.autobiographical=1 OR f.self_relevance>=0.5)
                ORDER BY COALESCE(f.event_start_at,f.valid_from,f.created_at),
                         e.ordinal,em.sequence_index
                LIMIT ?
                """,
                (max(1, min(1000, int(limit))),),
            ).fetchall()
        return [self.store._row_to_dict(row) for row in rows]

    def autonoetic_profile(self) -> dict[str, Any]:
        """Report operational self-recollection metadata, not consciousness."""
        timeline = self.autobiographical_timeline(limit=1000)
        remember = [m for m in timeline if m["recollection_mode"] == "remember"]
        return {
            "claim": "operational self-recollection metadata; not phenomenal consciousness",
            "autobiographical_memories": len(timeline),
            "remember_mode": len(remember),
            "field_perspective": sum(m["perspective"] == "field" for m in remember),
            "mean_vividness": round(
                sum(float(m["vividness"]) for m in remember) / len(remember), 4
            )
            if remember
            else 0.0,
        }

    def status(self) -> dict[str, Any]:
        with self.store._lock:
            counts = {
                "contexts": self.store._conn.execute(
                    "SELECT COUNT(*) FROM temporal_contexts"
                ).fetchone()[0],
                "events": self.store._conn.execute(
                    "SELECT COUNT(*) FROM episodic_events"
                ).fetchone()[0],
                "temporal_bindings": self.store._conn.execute(
                    "SELECT COUNT(*) FROM temporal_bindings"
                ).fetchone()[0],
                "source_assessments": self.store._conn.execute(
                    "SELECT COUNT(*) FROM source_monitoring_assessments"
                ).fetchone()[0],
                "labile_memories": self.store._conn.execute(
                    "SELECT COUNT(*) FROM reconsolidation_windows "
                    "WHERE status='labile' AND closes_at>CURRENT_TIMESTAMP"
                ).fetchone()[0],
            }
        return {**counts, **self.autonoetic_profile()}

    def backfill_existing(self, *, max_memories: int = 5000) -> dict[str, int]:
        """Idempotently enrich pre-cognitive memories with source/time structure."""
        limit = max(1, min(50_000, int(max_memories)))
        with self.store._lock:
            self.store._conn.execute(
                """
                UPDATE facts SET
                    event_start_at=COALESCE(event_start_at,valid_from,created_at),
                    event_end_at=COALESCE(
                        event_end_at,event_start_at,valid_from,created_at
                    ),
                    autobiographical=CASE WHEN memory_kind='identity' THEN 1
                                         ELSE autobiographical END,
                    self_relevance=CASE WHEN memory_kind='identity'
                                        THEN MAX(self_relevance,1.0)
                                        ELSE self_relevance END,
                    perspective=CASE WHEN memory_kind='identity' AND
                                          perspective='unknown'
                                     THEN 'semantic' ELSE perspective END,
                    recollection_mode=CASE WHEN memory_kind='identity'
                                               AND recollection_mode='know'
                                           THEN 'inferred'
                                           ELSE recollection_mode END
                """
            )
            rows = self.store._conn.execute(
                """
                SELECT fact_id,provenance_ref FROM facts
                ORDER BY COALESCE(event_start_at,valid_from,created_at),fact_id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        assessed = 0
        groups: dict[str, list[int]] = {}
        for row in rows:
            fact_id = int(row["fact_id"])
            with self.store._lock:
                exists = self.store._conn.execute(
                    "SELECT 1 FROM source_monitoring_assessments "
                    "WHERE fact_id=? LIMIT 1",
                    (fact_id,),
                ).fetchone()
            if not exists:
                self.monitor_source(fact_id)
                assessed += 1
            ref = str(row["provenance_ref"] or f"legacy:{fact_id}")
            groups.setdefault(ref, []).append(fact_id)
        segmented = 0
        for ref, ids in groups.items():
            with self.store._lock:
                unbound = self.store._conn.execute(
                    "SELECT COUNT(*) FROM facts WHERE fact_id IN ("
                    + ",".join("?" for _ in ids)
                    + ") AND context_id=''",
                    ids,
                ).fetchone()[0]
            if unbound:
                context_id = (
                    ref
                    if ref.startswith(("session:", "context:"))
                    else f"context:{hashlib.sha256(ref.encode()).hexdigest()[:20]}"
                )
                self.segment_memories(ids, context_id=context_id)
                segmented += len(ids)
        return {
            "memories_seen": len(rows),
            "source_assessments_created": assessed,
            "memories_segmented": segmented,
            "contexts_created_or_updated": len(groups),
        }

    def _snapshot(self, fact_id: int) -> dict[str, Any]:
        fact = self.store.get_fact(fact_id)
        if not fact:
            raise KeyError(f"fact_id {fact_id} not found")
        fact.pop("hrr_vector", None)
        return fact

    def _version(
        self,
        fact_id: int,
        window_id: str,
        phase: str,
        snapshot: dict[str, Any],
        reason: str,
    ) -> None:
        payload = _json(snapshot)
        self.store._conn.execute(
            "INSERT INTO memory_versions VALUES(?,?,?,?,?,?,?,?)",
            (
                str(uuid.uuid4()),
                fact_id,
                window_id,
                phase,
                payload,
                hashlib.sha256(payload.encode()).hexdigest(),
                reason,
                _now(),
            ),
        )
