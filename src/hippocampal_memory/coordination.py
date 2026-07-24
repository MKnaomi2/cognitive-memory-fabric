"""Immutable memory events and vault/neural synchronization ledgers."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from .store import MemoryStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_events (
    event_id TEXT PRIMARY KEY,
    aggregate_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    revision INTEGER NOT NULL,
    occurred_at TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    actor_ref TEXT NOT NULL DEFAULT '',
    source_uri TEXT NOT NULL DEFAULT '',
    causation_id TEXT,
    correlation_id TEXT,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    UNIQUE (aggregate_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_memory_events_aggregate
    ON memory_events(aggregate_id, revision);
CREATE INDEX IF NOT EXISTS idx_memory_events_correlation
    ON memory_events(correlation_id);
CREATE TABLE IF NOT EXISTS vault_registry (
    memory_id TEXT PRIMARY KEY,
    note_id TEXT NOT NULL UNIQUE,
    note_path TEXT NOT NULL UNIQUE,
    sync_revision INTEGER NOT NULL DEFAULT 0,
    memory_revision INTEGER NOT NULL DEFAULT 0,
    content_sha256 TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'registered',
    last_synced_at TEXT,
    last_error TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS sync_ledger (
    sync_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    from_revision INTEGER NOT NULL,
    to_revision INTEGER NOT NULL,
    note_path TEXT NOT NULL,
    before_sha256 TEXT NOT NULL DEFAULT '',
    after_sha256 TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}',
    occurred_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sync_ledger_memory
    ON sync_ledger(memory_id, occurred_at);
CREATE TABLE IF NOT EXISTS engram_bindings (
    memory_id TEXT PRIMARY KEY,
    engram_id TEXT NOT NULL UNIQUE,
    circuit_version TEXT NOT NULL,
    neuron_ids_json TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 0.5,
    replay_count INTEGER NOT NULL DEFAULT 0,
    last_replayed_at TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS neural_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    circuit_version TEXT NOT NULL,
    phase TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    event_revision INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS time_cell_bindings (
    memory_id TEXT PRIMARY KEY,
    context_id TEXT NOT NULL DEFAULT '',
    event_id TEXT NOT NULL DEFAULT '',
    sequence_index INTEGER,
    cell_ids_json TEXT NOT NULL,
    preferred_phase_json TEXT NOT NULL,
    circuit_version TEXT NOT NULL,
    last_replayed_at TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_time_cell_context_sequence
    ON time_cell_bindings(context_id,event_id,sequence_index);
"""

_ACTORS = {"user", "agent", "web", "reflection", "sensor", "system", "import"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    aggregate_id: str
    event_type: str
    schema_version: int
    revision: int
    occurred_at: str
    actor_type: str
    actor_ref: str
    source_uri: str
    causation_id: str | None
    correlation_id: str | None
    payload: dict[str, Any]
    payload_sha256: str


class RevisionConflict(RuntimeError):
    """A write was attempted against a stale aggregate revision."""


class MemoryCoordinator:
    """Coordinate the current-state store with append-only integration state."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        with store._lock:
            store._conn.executescript(_SCHEMA)
            columns = {
                row[1]
                for row in store._conn.execute(
                    "PRAGMA table_info(engram_bindings)"
                ).fetchall()
            }
            for name, declaration in {
                "encoding_version": "TEXT NOT NULL DEFAULT 'memory-id-v2'",
                "content_sha256": "TEXT NOT NULL DEFAULT ''",
                "ca1_signature_json": "TEXT NOT NULL DEFAULT '[]'",
            }.items():
                if name not in columns:
                    store._conn.execute(
                        f"ALTER TABLE engram_bindings ADD COLUMN {name} {declaration}"
                    )

    def current_revision(self, aggregate_id: str) -> int:
        with self.store._lock:
            row = self.store._conn.execute(
                "SELECT COALESCE(MAX(revision), 0) revision FROM memory_events "
                "WHERE aggregate_id = ?",
                (aggregate_id,),
            ).fetchone()
            return int(row["revision"])

    def append_event(
        self,
        aggregate_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        actor_type: str,
        actor_ref: str = "",
        source_uri: str = "",
        expected_revision: int | None = None,
        causation_id: str | None = None,
        correlation_id: str | None = None,
        event_id: str | None = None,
        occurred_at: str | None = None,
    ) -> MemoryEvent:
        """Append an idempotent event with optimistic concurrency control."""
        if actor_type not in _ACTORS:
            raise ValueError(f"invalid actor_type: {actor_type}")
        aggregate_id, event_type = aggregate_id.strip(), event_type.strip()
        if not aggregate_id or not event_type:
            raise ValueError("aggregate_id and event_type are required")
        event_id = event_id or str(uuid.uuid4())
        payload_json = _json(payload)
        digest = hashlib.sha256(payload_json.encode()).hexdigest()
        with self.store._lock:
            row = self.store._conn.execute(
                "SELECT * FROM memory_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if row:
                if (
                    row["aggregate_id"] != aggregate_id
                    or row["event_type"] != event_type
                    or row["payload_sha256"] != digest
                ):
                    raise RevisionConflict("event_id already identifies another event")
                return self._event(row)
            current = self.current_revision(aggregate_id)
            if expected_revision is not None and current != expected_revision:
                raise RevisionConflict(
                    f"expected revision {expected_revision}, found {current}"
                )
            self.store._conn.execute(
                """
                INSERT INTO memory_events (
                    event_id, aggregate_id, event_type, schema_version, revision,
                    occurred_at, actor_type, actor_ref, source_uri, causation_id,
                    correlation_id, payload_json, payload_sha256
                ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    aggregate_id,
                    event_type,
                    current + 1,
                    occurred_at or _now(),
                    actor_type,
                    actor_ref,
                    source_uri,
                    causation_id,
                    correlation_id,
                    payload_json,
                    digest,
                ),
            )
            row = self.store._conn.execute(
                "SELECT * FROM memory_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            return self._event(row)

    def events(self, aggregate_id: str, after_revision: int = 0) -> list[MemoryEvent]:
        with self.store._lock:
            rows = self.store._conn.execute(
                "SELECT * FROM memory_events WHERE aggregate_id = ? "
                "AND revision > ? ORDER BY revision",
                (aggregate_id, after_revision),
            ).fetchall()
            return [self._event(row) for row in rows]

    def ingest(
        self,
        content: str,
        *,
        actor_type: str,
        actor_ref: str = "",
        source_uri: str = "",
        provenance: dict[str, Any] | None = None,
        **memory_fields: Any,
    ) -> int:
        """Create/reobserve memory state and its provenance-bearing event."""
        fact_id = self.store.add_fact(
            content,
            provenance_type=actor_type if actor_type != "import" else "imported",
            provenance_ref=actor_ref or source_uri,
            provenance={
                **(provenance or {}),
                "actor_ref": actor_ref,
                "source_uri": source_uri,
            },
            **memory_fields,
        )
        self.append_event(
            f"memory:{fact_id}",
            "memory.observed",
            {
                "memory_id": fact_id,
                "content_sha256": hashlib.sha256(
                    content.strip().encode()
                ).hexdigest(),
                "provenance": provenance or {},
            },
            actor_type=actor_type,
            actor_ref=actor_ref,
            source_uri=source_uri,
        )
        return fact_id

    def bind_engram(
        self,
        memory_id: int | str,
        neuron_ids: Iterable[int],
        *,
        circuit_version: str,
        engram_id: str | None = None,
        strength: float = 0.5,
        encoding_version: str = "memory-id-v2",
        content_sha256: str = "",
        ca1_signature: Iterable[int] = (),
    ) -> str:
        memory_id, engram_id = str(memory_id), engram_id or str(uuid.uuid4())
        ids = sorted({int(value) for value in neuron_ids})
        ca1_ids = sorted({int(value) for value in ca1_signature})
        strength = max(0.0, min(1.0, float(strength)))
        with self.store._lock:
            self.store._conn.execute(
                """
                INSERT INTO engram_bindings (
                    memory_id, engram_id, circuit_version, neuron_ids_json,
                    strength, created_at, encoding_version, content_sha256,
                    ca1_signature_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    engram_id=excluded.engram_id,
                    circuit_version=excluded.circuit_version,
                    neuron_ids_json=excluded.neuron_ids_json,
                    strength=excluded.strength,
                    encoding_version=excluded.encoding_version,
                    content_sha256=excluded.content_sha256,
                    ca1_signature_json=excluded.ca1_signature_json
                """,
                (
                    memory_id,
                    engram_id,
                    circuit_version,
                    _json(ids),
                    strength,
                    _now(),
                    encoding_version,
                    content_sha256,
                    _json(ca1_ids),
                ),
            )
        self.append_event(
            f"memory:{memory_id}",
            "engram.bound",
            {
                "engram_id": engram_id,
                "circuit_version": circuit_version,
                "neuron_count": len(ids),
                "strength": strength,
                "encoding_version": encoding_version,
                "ca1_signature_count": len(ca1_ids),
            },
            actor_type="system",
            actor_ref="hippocampal-circuit",
        )
        return engram_id

    def bind_time_cells(
        self,
        memory_id: int | str,
        cell_ids: Iterable[int],
        *,
        preferred_phases: Iterable[float],
        circuit_version: str,
        context_id: str = "",
        event_id: str = "",
        sequence_index: int | None = None,
    ) -> None:
        """Bind a memory's temporal position to an inspectable time-cell assembly."""
        memory_id = str(memory_id)
        ids = [int(value) for value in cell_ids]
        phases = [max(0.0, min(1.0, float(value))) for value in preferred_phases]
        if not ids or len(ids) != len(phases):
            raise ValueError("time-cell IDs and preferred phases must align")
        with self.store._lock:
            self.store._conn.execute(
                """
                INSERT INTO time_cell_bindings(
                    memory_id,context_id,event_id,sequence_index,cell_ids_json,
                    preferred_phase_json,circuit_version,created_at
                ) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    context_id=excluded.context_id,event_id=excluded.event_id,
                    sequence_index=excluded.sequence_index,
                    cell_ids_json=excluded.cell_ids_json,
                    preferred_phase_json=excluded.preferred_phase_json,
                    circuit_version=excluded.circuit_version
                """,
                (
                    memory_id,
                    context_id,
                    event_id,
                    sequence_index,
                    _json(ids),
                    _json(phases),
                    circuit_version,
                    _now(),
                ),
            )
        self.append_event(
            f"memory:{memory_id}",
            "time-cells.bound",
            {
                "cell_count": len(ids),
                "context_id": context_id,
                "event_id": event_id,
                "sequence_index": sequence_index,
                "circuit_version": circuit_version,
            },
            actor_type="system",
            actor_ref="hippocampal-time-cells",
        )

    def register_vault_note(
        self, memory_id: int | str, note_path: str, note_id: str | None = None
    ) -> str:
        note_id = note_id or str(uuid.uuid4())
        with self.store._lock:
            self.store._conn.execute(
                "INSERT INTO vault_registry(memory_id,note_id,note_path) "
                "VALUES(?,?,?) ON CONFLICT(memory_id) DO UPDATE SET "
                "note_path=excluded.note_path",
                (str(memory_id), note_id, note_path.replace("\\", "/")),
            )
            row = self.store._conn.execute(
                "SELECT note_id FROM vault_registry WHERE memory_id=?",
                (str(memory_id),),
            ).fetchone()
            return str(row["note_id"])

    def record_sync(
        self,
        memory_id: int | str,
        *,
        direction: str,
        note_path: str,
        from_revision: int,
        to_revision: int,
        before_sha256: str,
        after_sha256: str,
        outcome: str,
        detail: dict[str, Any] | None = None,
    ) -> str:
        if direction not in {"memory_to_vault", "vault_to_memory", "reconcile"}:
            raise ValueError("invalid sync direction")
        sync_id, timestamp = str(uuid.uuid4()), _now()
        with self.store._lock:
            self.store._conn.execute(
                """
                INSERT INTO sync_ledger VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    sync_id,
                    str(memory_id),
                    direction,
                    from_revision,
                    to_revision,
                    note_path.replace("\\", "/"),
                    before_sha256,
                    after_sha256,
                    outcome,
                    _json(detail or {}),
                    timestamp,
                ),
            )
            if outcome == "applied":
                self.store._conn.execute(
                    "UPDATE vault_registry SET sync_revision=?,memory_revision=?,"
                    "content_sha256=?,state='synced',last_synced_at=?,last_error='' "
                    "WHERE memory_id=?",
                    (
                        to_revision,
                        to_revision,
                        after_sha256,
                        timestamp,
                        str(memory_id),
                    ),
                )
        return sync_id

    @staticmethod
    def _event(row: Any) -> MemoryEvent:
        return MemoryEvent(
            event_id=str(row["event_id"]),
            aggregate_id=str(row["aggregate_id"]),
            event_type=str(row["event_type"]),
            schema_version=int(row["schema_version"]),
            revision=int(row["revision"]),
            occurred_at=str(row["occurred_at"]),
            actor_type=str(row["actor_type"]),
            actor_ref=str(row["actor_ref"]),
            source_uri=str(row["source_uri"]),
            causation_id=row["causation_id"],
            correlation_id=row["correlation_id"],
            payload=json.loads(row["payload_json"]),
            payload_sha256=str(row["payload_sha256"]),
        )
