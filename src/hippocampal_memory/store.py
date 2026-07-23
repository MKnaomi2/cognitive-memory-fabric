"""
SQLite-backed fact store with entity resolution and trust scoring.
Standalone single-user memory lifecycle store.
"""

import json
import os
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_evidence (
    evidence_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id        INTEGER NOT NULL REFERENCES facts(fact_id),
    polarity       TEXT NOT NULL,
    source_type    TEXT NOT NULL,
    source_ref     TEXT DEFAULT '',
    detail         TEXT DEFAULT '',
    weight         REAL DEFAULT 1.0,
    observed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_conflicts (
    conflict_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_a_id      INTEGER NOT NULL REFERENCES facts(fact_id),
    fact_b_id      INTEGER NOT NULL REFERENCES facts(fact_id),
    reason         TEXT DEFAULT '',
    status         TEXT DEFAULT 'open',
    detected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at    TIMESTAMP,
    UNIQUE(fact_a_id, fact_b_id)
);

CREATE TABLE IF NOT EXISTS fact_derivations (
    derived_fact_id INTEGER NOT NULL REFERENCES facts(fact_id),
    source_fact_id  INTEGER NOT NULL REFERENCES facts(fact_id),
    relation        TEXT DEFAULT 'consolidated_from',
    PRIMARY KEY (derived_fact_id, source_fact_id)
);

CREATE TABLE IF NOT EXISTS hippocampus_sessions (
    session_id       TEXT PRIMARY KEY,
    last_message_id  INTEGER NOT NULL DEFAULT 0,
    status           TEXT NOT NULL DEFAULT 'queued',
    attempts         INTEGER NOT NULL DEFAULT 0,
    last_error       TEXT DEFAULT '',
    eligible_at      TIMESTAMP,
    processed_at     TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hippocampus_runs (
    run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    mode              TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt_version    TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'running',
    sessions_seen     INTEGER NOT NULL DEFAULT 0,
    memories_created  INTEGER NOT NULL DEFAULT 0,
    consolidated      INTEGER NOT NULL DEFAULT 0,
    superseded        INTEGER NOT NULL DEFAULT 0,
    archived          INTEGER NOT NULL DEFAULT 0,
    rejected          INTEGER NOT NULL DEFAULT 0,
    error             TEXT DEFAULT '',
    started_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at       TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memory_decisions (
    decision_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER REFERENCES hippocampus_runs(run_id),
    action         TEXT NOT NULL,
    target_fact_id INTEGER,
    source_ids     TEXT DEFAULT '[]',
    reason         TEXT NOT NULL,
    accepted       INTEGER NOT NULL,
    payload_json   TEXT DEFAULT '{}',
    decided_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hippocampus_control (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _apply_wal_with_fallback(conn: sqlite3.Connection) -> None:
    """Prefer WAL, falling back to the portable rollback journal."""
    try:
        mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            conn.execute("PRAGMA journal_mode=DELETE")
    except sqlite3.DatabaseError:
        conn.execute("PRAGMA journal_mode=DELETE")


# Trust adjustment constants
_HELPFUL_DELTA = 0.05
_UNHELPFUL_DELTA = -0.10
_TRUST_MIN = 0.0
_TRUST_MAX = 1.0
_SOURCE_TYPES = {"user", "agent", "web", "reflection", "sensor", "system", "imported"}
_MEMORY_KINDS = {"episode", "fact", "principle", "identity"}

# Entity extraction patterns
_RE_CAPITALIZED = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")
_RE_AKA = re.compile(
    r"(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)",
    re.IGNORECASE,
)


def _clamp_trust(value: float) -> float:
    return max(_TRUST_MIN, min(_TRUST_MAX, value))


class MemoryStore:
    """SQLite-backed fact store with entity resolution and trust scoring."""

    # --- Process-wide shared connection registry -------------------------
    # SQLite permits only one writer at a time. Each MemoryStore instance used
    # to open its own connection guarded by its own RLock, so the several
    # providers that coexist in one process (the main agent plus every
    # delegate_task subagent) raced as independent WAL writers. Combined with
    # writes that were not rolled back on error, one connection could leave an
    # open write transaction that pinned the write lock and made every other
    # connection's write fail with "database is locked" for the full busy
    # timeout. All instances for the same database now share ONE connection and
    # ONE re-entrant lock, so access is fully serialized and cross-connection
    # contention is impossible. The shared connection is refcounted, so closing
    # one instance never tears the connection out from under a live sibling.
    _shared: dict = {}
    _shared_guard = threading.Lock()

    def __init__(
        self,
        db_path: "str | Path | None" = None,
        default_trust: float = 0.5,
        hrr_dim: int = 4096,
    ) -> None:
        if db_path is None:
            configured = os.environ.get("HIPPOCAMPAL_MEMORY_HOME")
            home = (
                Path(configured).expanduser()
                if configured
                else Path.home() / ".hippocampal-memory"
            )
            db_path = str(home / "memory_store.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_trust = _clamp_trust(default_trust)
        self.hrr_dim = hrr_dim
        self._hrr_available = hrr._HAS_NUMPY

        # Acquire (or open) the process-wide shared connection for this DB.
        # resolve() (not just expanduser) so symlinked/relative paths to the
        # same file share ONE connection instead of silently reintroducing
        # the multi-writer contention this registry exists to prevent.
        try:
            self._key = str(self.db_path.resolve())
        except OSError:
            self._key = str(self.db_path)
        with MemoryStore._shared_guard:
            entry = MemoryStore._shared.get(self._key)
            if entry is None:
                conn = sqlite3.connect(
                    self._key,
                    check_same_thread=False,
                    timeout=10.0,
                    # Autocommit: every statement is its own transaction, so a
                    # write that raises mid-method can never leave a dangling
                    # transaction (and its write lock) open. The explicit
                    # commit() calls below become harmless no-ops.
                    isolation_level=None,
                )
                conn.row_factory = sqlite3.Row
                entry = {
                    "conn": conn,
                    "lock": threading.RLock(),
                    "refs": 0,
                    "ready": False,
                }
                MemoryStore._shared[self._key] = entry
            entry["refs"] += 1
            self._entry = entry
            self._conn = entry["conn"]
            self._lock = entry["lock"]

        # Initialise the schema once per shared connection.
        with self._lock:
            if not self._entry["ready"]:
                self._init_db()
                self._entry["ready"] = True

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables, indexes, and triggers if they do not exist. Enable WAL mode."""
        _apply_wal_with_fallback(self._conn)
        self._conn.executescript(_SCHEMA)
        # Migrate: add hrr_vector column if missing (safe for existing databases)
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()
        }
        migrations = {
            "hrr_vector": "BLOB",
            "memory_kind": "TEXT NOT NULL DEFAULT 'fact'",
            "status": "TEXT NOT NULL DEFAULT 'active'",
            "provenance_type": "TEXT NOT NULL DEFAULT 'imported'",
            "provenance_ref": "TEXT DEFAULT ''",
            "provenance_json": "TEXT DEFAULT '{}'",
            "confirmation_count": "INTEGER NOT NULL DEFAULT 0",
            "contradiction_count": "INTEGER NOT NULL DEFAULT 0",
            "last_confirmed_at": "TIMESTAMP",
            "archived_at": "TIMESTAMP",
            "subject_key": "TEXT DEFAULT ''",
            "predicate_key": "TEXT DEFAULT ''",
            "valid_from": "TIMESTAMP",
            "valid_until": "TIMESTAMP",
            "expires_at": "TIMESTAMP",
            "last_accessed_at": "TIMESTAMP",
            "relevance_score": "REAL NOT NULL DEFAULT 0.5",
            "salience_score": "REAL NOT NULL DEFAULT 0.5",
            "source_quality": "REAL NOT NULL DEFAULT 0.5",
            "pinned": "INTEGER NOT NULL DEFAULT 0",
            "superseded_by": "INTEGER",
            "archive_reason": "TEXT DEFAULT ''",
            "review_after": "TIMESTAMP",
            "restored_at": "TIMESTAMP",
        }
        for name, declaration in migrations.items():
            if name not in columns:
                self._conn.execute(f"ALTER TABLE facts ADD COLUMN {name} {declaration}")
        self._conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);
            CREATE INDEX IF NOT EXISTS idx_facts_kind ON facts(memory_kind);
            CREATE INDEX IF NOT EXISTS idx_evidence_fact ON fact_evidence(fact_id, observed_at);
            CREATE INDEX IF NOT EXISTS idx_conflicts_status ON fact_conflicts(status);
            CREATE INDEX IF NOT EXISTS idx_facts_subject_predicate
                ON facts(subject_key, predicate_key);
            CREATE INDEX IF NOT EXISTS idx_facts_review_after ON facts(review_after);
            CREATE INDEX IF NOT EXISTS idx_hippocampus_sessions_status
                ON hippocampus_sessions(status, eligible_at);
            INSERT OR IGNORE INTO hippocampus_control(key, value)
                VALUES ('paused', 'false');
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        category: str = "general",
        tags: str = "",
        *,
        provenance_type: str = "agent",
        provenance_ref: str = "",
        provenance: dict | None = None,
        confidence: float | None = None,
        memory_kind: str = "fact",
        derived_from: list[int] | None = None,
        subject_key: str = "",
        predicate_key: str = "",
        valid_from: str | None = None,
        valid_until: str | None = None,
        expires_at: str | None = None,
        relevance_score: float = 0.5,
        salience_score: float = 0.5,
        source_quality: float = 0.5,
        pinned: bool = False,
    ) -> int:
        """Insert a fact and return its fact_id.

        Every memory records provenance and confidence. Re-observing identical
        content adds confirming evidence and raises confidence instead of
        silently returning an unchanged duplicate.
        """
        with self._lock:
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")
            if provenance_type not in _SOURCE_TYPES:
                raise ValueError(f"invalid provenance_type: {provenance_type}")
            if memory_kind not in _MEMORY_KINDS:
                raise ValueError(f"invalid memory_kind: {memory_kind}")
            initial_confidence = (
                self.default_trust
                if confidence is None
                else _clamp_trust(float(confidence))
            )
            provenance_json = json.dumps(provenance or {}, sort_keys=True, default=str)

            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO facts (
                        content, category, tags, trust_score, memory_kind,
                        provenance_type, provenance_ref, provenance_json,
                        subject_key, predicate_key, valid_from, valid_until,
                        expires_at, relevance_score, salience_score,
                        source_quality, pinned
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        content,
                        category,
                        tags,
                        initial_confidence,
                        memory_kind,
                        provenance_type,
                        provenance_ref,
                        provenance_json,
                        subject_key.strip().lower(),
                        predicate_key.strip().lower(),
                        valid_from,
                        valid_until,
                        expires_at,
                        _clamp_trust(float(relevance_score)),
                        _clamp_trust(float(salience_score)),
                        _clamp_trust(float(source_quality)),
                        int(bool(pinned)),
                    ),
                )
                self._conn.commit()
                fact_id: int = cur.lastrowid  # type: ignore[assignment]
            except sqlite3.IntegrityError:
                # Duplicate content is later corroboration, not a no-op.
                row = self._conn.execute(
                    "SELECT fact_id FROM facts WHERE content = ?", (content,)
                ).fetchone()
                fact_id = int(row["fact_id"])
                self.record_evidence(
                    fact_id,
                    "confirm",
                    provenance_type=provenance_type,
                    provenance_ref=provenance_ref,
                    detail="Repeated observation",
                    observed_at=valid_from,
                )
                return fact_id

            self._conn.execute(
                """
                INSERT INTO fact_evidence
                    (fact_id, polarity, source_type, source_ref, detail, weight, observed_at)
                VALUES (?, 'source', ?, ?, ?, 1.0, COALESCE(?, CURRENT_TIMESTAMP))
                """,
                (fact_id, provenance_type, provenance_ref, provenance_json, valid_from),
            )
            for source_id in derived_from or []:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO fact_derivations
                        (derived_fact_id, source_fact_id)
                    VALUES (?, ?)
                    """,
                    (fact_id, int(source_id)),
                )

            # Entity extraction and linking
            for name in self._extract_entities(content):
                entity_id = self._resolve_entity(name)
                self._link_fact_entity(fact_id, entity_id)

            # Compute HRR vector after entity linking
            self._compute_hrr_vector(fact_id, content)
            self._rebuild_bank(category)

            return fact_id

    def get_fact(self, fact_id: int) -> dict | None:
        """Return one memory with its lifecycle metadata."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def record_evidence(
        self,
        fact_id: int,
        polarity: str,
        *,
        provenance_type: str,
        provenance_ref: str = "",
        detail: str = "",
        weight: float = 1.0,
        observed_at: str | None = None,
    ) -> dict:
        """Attach evidence and update confidence with bounded accumulation."""
        if polarity not in {"source", "confirm", "contradict"}:
            raise ValueError("polarity must be source, confirm, or contradict")
        if provenance_type not in _SOURCE_TYPES:
            raise ValueError(f"invalid provenance_type: {provenance_type}")
        weight = max(0.0, min(1.0, float(weight)))
        with self._lock:
            row = self._conn.execute(
                "SELECT trust_score, confirmation_count, contradiction_count FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"fact_id {fact_id} not found")
            old = float(row["trust_score"])
            new = old
            confirmation_inc = contradiction_inc = 0
            if polarity == "confirm":
                new = _clamp_trust(old + (1.0 - old) * 0.15 * weight)
                confirmation_inc = 1
            elif polarity == "contradict":
                new = _clamp_trust(old - old * 0.25 * weight)
                contradiction_inc = 1
            self._conn.execute(
                """
                INSERT INTO fact_evidence
                    (fact_id, polarity, source_type, source_ref, detail, weight)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (fact_id, polarity, provenance_type, provenance_ref, detail, weight),
            )
            if observed_at:
                self._conn.execute(
                    """
                    UPDATE fact_evidence SET observed_at = ?
                    WHERE evidence_id = last_insert_rowid()
                    """,
                    (observed_at,),
                )
            self._conn.execute(
                """
                UPDATE facts SET trust_score = ?,
                    confirmation_count = confirmation_count + ?,
                    contradiction_count = contradiction_count + ?,
                    last_confirmed_at = CASE WHEN ? = 1 THEN CURRENT_TIMESTAMP ELSE last_confirmed_at END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE fact_id = ?
                """,
                (new, confirmation_inc, contradiction_inc, confirmation_inc, fact_id),
            )
            return {
                "fact_id": fact_id,
                "old_confidence": old,
                "confidence": new,
                "polarity": polarity,
            }

    def search_facts(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Full-text search over facts using FTS5.

        Returns a list of fact dicts ordered by FTS5 rank, then trust_score
        descending. Also increments retrieval_count for matched facts.
        """
        with self._lock:
            query = query.strip()
            if not query:
                return []

            # FTS5 AND-joins tokens by default, which zeroes out recall on
            # natural-language queries. Reuse the retriever's sanitizer
            # (stopword drop + OR-join content tokens). Imported lazily to
            # avoid a store->retrieval import cycle.
            from .retrieval import FactRetriever

            match_query = FactRetriever._sanitize_fts_query(query)
            params: list = [match_query, min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND f.category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT f.fact_id, f.content, f.category, f.tags,
                       f.trust_score, f.retrieval_count, f.helpful_count,
                       f.created_at, f.updated_at, f.memory_kind, f.status,
                       f.provenance_type, f.provenance_ref, f.provenance_json,
                       f.confirmation_count, f.contradiction_count,
                       f.last_confirmed_at, f.archived_at
                FROM facts f
                JOIN facts_fts fts ON fts.rowid = f.fact_id
                WHERE facts_fts MATCH ?
                  AND f.trust_score >= ?
                  AND f.status != 'archived'
                  {category_clause}
                ORDER BY fts.rank, f.trust_score DESC
                LIMIT ?
            """

            rows = self._conn.execute(sql, params).fetchall()
            results = [self._row_to_dict(r) for r in rows]

            if results:
                ids = [r["fact_id"] for r in results]
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(
                    f"UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})",
                    ids,
                )
                self._conn.commit()

            return results

    def update_fact(
        self,
        fact_id: int,
        content: str | None = None,
        trust_delta: float | None = None,
        tags: str | None = None,
        category: str | None = None,
    ) -> bool:
        """Partially update a fact. Trust is clamped to [0, 1].

        Returns True if the row existed, False otherwise.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            assignments: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
            params: list = []

            if content is not None:
                assignments.append("content = ?")
                params.append(content.strip())
            if tags is not None:
                assignments.append("tags = ?")
                params.append(tags)
            if category is not None:
                assignments.append("category = ?")
                params.append(category)
            if trust_delta is not None:
                new_trust = _clamp_trust(row["trust_score"] + trust_delta)
                assignments.append("trust_score = ?")
                params.append(new_trust)

            params.append(fact_id)
            self._conn.execute(
                f"UPDATE facts SET {', '.join(assignments)} WHERE fact_id = ?",
                params,
            )
            self._conn.commit()

            # If content changed, re-extract entities
            if content is not None:
                self._conn.execute(
                    "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
                )
                for name in self._extract_entities(content):
                    entity_id = self._resolve_entity(name)
                    self._link_fact_entity(fact_id, entity_id)
                self._conn.commit()

            # Recompute HRR vector if content changed
            if content is not None:
                self._compute_hrr_vector(fact_id, content)
            # Rebuild bank for relevant category
            cat = (
                category
                or self._conn.execute(
                    "SELECT category FROM facts WHERE fact_id = ?", (fact_id,)
                ).fetchone()["category"]
            )
            self._rebuild_bank(cat)

            return True

    def remove_fact(self, fact_id: int) -> bool:
        """Archive a fact so normal retrieval forgets it without erasing it."""
        return self.archive_fact(fact_id)

    def archive_fact(self, fact_id: int, reason: str = "manual") -> bool:
        """Make a memory harder to retrieve while preserving all evidence."""
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, category FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            self._conn.execute(
                """
                UPDATE facts SET status = 'archived', archived_at = CURRENT_TIMESTAMP,
                    archive_reason = ?,
                    updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?
                """,
                (reason[:500], fact_id),
            )
            self._conn.commit()
            self._rebuild_bank(row["category"])
            return True

    def restore_fact(self, fact_id: int) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT category FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False
            self._conn.execute(
                """
                UPDATE facts SET status = 'active', archived_at = NULL,
                    archive_reason = '', restored_at = CURRENT_TIMESTAMP,
                    review_after = datetime('now', '+30 days'),
                    updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?
                """,
                (fact_id,),
            )
            self._rebuild_bank(row["category"])
            return True

    def consolidate(
        self,
        content: str,
        source_fact_ids: list[int],
        *,
        memory_kind: str = "principle",
        provenance_type: str = "reflection",
        provenance_ref: str = "",
        archive_sources: bool = True,
        category: str = "general",
    ) -> int:
        """Create a reusable principle/identity memory with a traceable lineage."""
        if memory_kind not in {"principle", "identity"}:
            raise ValueError("consolidated memory_kind must be principle or identity")
        if not source_fact_ids:
            raise ValueError("source_fact_ids must not be empty")
        with self._lock:
            placeholders = ",".join("?" * len(source_fact_ids))
            count = self._conn.execute(
                f"SELECT COUNT(*) FROM facts WHERE fact_id IN ({placeholders})",
                source_fact_ids,
            ).fetchone()[0]
            if count != len(set(source_fact_ids)):
                raise KeyError("one or more source facts do not exist")
            fact_id = self.add_fact(
                content,
                category=category,
                provenance_type=provenance_type,
                provenance_ref=provenance_ref,
                memory_kind=memory_kind,
                derived_from=source_fact_ids,
                provenance={
                    "operation": "consolidation",
                    "source_fact_ids": source_fact_ids,
                },
            )
            if archive_sources:
                for source_id in source_fact_ids:
                    if source_id != fact_id:
                        self.archive_fact(source_id)
            return fact_id

    def schedule_consolidated_sources(
        self, derived_fact_id: int, source_fact_ids: list[int], grace_days: int = 7
    ) -> None:
        """Schedule source episodes for reversible archival after a grace period."""
        if not source_fact_ids:
            return
        with self._lock:
            placeholders = ",".join("?" * len(source_fact_ids))
            self._conn.execute(
                f"""
                UPDATE facts
                SET review_after = datetime('now', ?),
                    updated_at = CURRENT_TIMESTAMP
                WHERE fact_id IN ({placeholders})
                  AND memory_kind = 'episode'
                  AND status != 'archived'
                """,
                [f"+{max(1, int(grace_days))} days", *source_fact_ids],
            )
            self._conn.execute(
                """
                INSERT INTO memory_decisions
                    (action, target_fact_id, source_ids, reason, accepted)
                VALUES ('consolidation_grace', ?, ?, ?, 1)
                """,
                (
                    derived_fact_id,
                    json.dumps(source_fact_ids),
                    f"Source episodes retained for {grace_days}-day consolidation grace",
                ),
            )

    def supersede_fact(self, old_fact_id: int, new_fact_id: int, reason: str) -> bool:
        """Archive an older state while preserving its historical validity."""
        if old_fact_id == new_fact_id:
            return False
        with self._lock:
            old = self._conn.execute(
                "SELECT * FROM facts WHERE fact_id = ?", (old_fact_id,)
            ).fetchone()
            new = self._conn.execute(
                "SELECT * FROM facts WHERE fact_id = ?", (new_fact_id,)
            ).fetchone()
            if old is None or new is None:
                return False
            if not old["subject_key"] or not old["predicate_key"]:
                return False
            if (
                old["subject_key"] != new["subject_key"]
                or old["predicate_key"] != new["predicate_key"]
                or float(new["source_quality"]) < float(old["source_quality"])
            ):
                return False
            self._conn.execute(
                """
                UPDATE facts SET status = 'archived', archived_at = CURRENT_TIMESTAMP,
                    archive_reason = ?, superseded_by = ?,
                    valid_until = COALESCE(valid_until, CURRENT_TIMESTAMP),
                    updated_at = CURRENT_TIMESTAMP
                WHERE fact_id = ?
                """,
                (f"superseded: {reason}"[:500], new_fact_id, old_fact_id),
            )
            return True

    def assess_consolidation(
        self, source_fact_ids: list[int], memory_kind: str
    ) -> dict:
        """Apply conservative, deterministic evidence thresholds."""
        if memory_kind not in {"principle", "identity"} or not source_fact_ids:
            return {"eligible": False, "reason": "invalid kind or empty evidence"}
        with self._lock:
            placeholders = ",".join("?" * len(source_fact_ids))
            rows = self._conn.execute(
                f"""
                SELECT fact_id, trust_score, provenance_ref,
                       COALESCE(valid_from, created_at) AS evidence_at, status
                FROM facts WHERE fact_id IN ({placeholders})
                  AND memory_kind IN ('episode', 'fact')
                """,
                source_fact_ids,
            ).fetchall()
            refs = {
                str(row["provenance_ref"] or "")
                for row in rows
                if row["provenance_ref"]
            }
            evidence_rows = self._conn.execute(
                f"""
                SELECT source_ref, observed_at FROM fact_evidence
                WHERE fact_id IN ({placeholders})
                  AND polarity IN ('source', 'confirm')
                """,
                source_fact_ids,
            ).fetchall()
            refs.update(
                str(row["source_ref"]) for row in evidence_rows if row["source_ref"]
            )
            mean_conf = (
                sum(float(row["trust_score"]) for row in rows) / len(rows)
                if rows
                else 0.0
            )
            open_conflicts = self._conn.execute(
                f"""
                SELECT COUNT(*) FROM fact_conflicts
                WHERE status = 'open'
                  AND (fact_a_id IN ({placeholders}) OR fact_b_id IN ({placeholders}))
                """,
                [*source_fact_ids, *source_fact_ids],
            ).fetchone()[0]
            contradiction_count = self._conn.execute(
                f"""
                SELECT COALESCE(SUM(contradiction_count), 0)
                FROM facts WHERE fact_id IN ({placeholders})
                """,
                source_fact_ids,
            ).fetchone()[0]
            confirmation_count = self._conn.execute(
                f"""
                SELECT COALESCE(SUM(confirmation_count), 0)
                FROM facts WHERE fact_id IN ({placeholders})
                """,
                source_fact_ids,
            ).fetchone()[0]
            minimum = 5 if memory_kind == "identity" else 3
            min_refs = 3 if memory_kind == "identity" else 2
            min_conf = 0.80 if memory_kind == "identity" else 0.70
            ratio = contradiction_count / max(
                1, contradiction_count + confirmation_count + len(rows)
            )
            span_days = 0.0
            raw_stamps = [row["evidence_at"] for row in rows]
            raw_stamps.extend(row["observed_at"] for row in evidence_rows)
            stamps = []
            for raw in raw_stamps:
                if not raw:
                    continue
                parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                if parsed.tzinfo is not None:
                    parsed = parsed.replace(tzinfo=None)
                stamps.append(parsed)
            if stamps:
                span_days = (max(stamps) - min(stamps)).total_seconds() / 86400
            support_count = max(len(rows), len(refs))
            eligible = (
                support_count >= minimum
                and len(refs) >= min_refs
                and mean_conf >= min_conf
                and open_conflicts == 0
                and ratio < 0.20
                and (memory_kind != "identity" or span_days >= 7)
            )
            return {
                "eligible": eligible,
                "evidence_count": support_count,
                "independent_sources": len(refs),
                "mean_confidence": round(mean_conf, 4),
                "contradiction_ratio": round(ratio, 4),
                "span_days": round(span_days, 2),
                "open_conflicts": int(open_conflicts),
                "reason": "thresholds satisfied"
                if eligible
                else "conservative thresholds not satisfied",
            }

    def run_forgetting_maintenance(self) -> dict:
        """Archive only deterministic, reversible forgetting candidates."""
        with self._lock:
            archived: list[int] = []
            rules = [
                (
                    "expired",
                    """expires_at IS NOT NULL AND expires_at <= CURRENT_TIMESTAMP
                       AND pinned = 0""",
                ),
                (
                    "consolidated-grace-complete",
                    """memory_kind = 'episode' AND review_after IS NOT NULL
                       AND review_after <= CURRENT_TIMESTAMP AND pinned = 0
                       AND EXISTS (
                         SELECT 1 FROM fact_derivations d
                         WHERE d.source_fact_id = facts.fact_id
                       )""",
                ),
                (
                    "stale-low-value",
                    """memory_kind = 'episode' AND created_at <= datetime('now', '-180 days')
                       AND trust_score < 0.45 AND retrieval_count = 0 AND pinned = 0
                       AND review_after IS NULL
                       AND NOT EXISTS (
                         SELECT 1 FROM fact_conflicts c
                         WHERE c.status = 'open'
                           AND (c.fact_a_id = facts.fact_id OR c.fact_b_id = facts.fact_id)
                       )""",
                ),
                (
                    "resolved-low-confidence",
                    """trust_score < 0.25 AND contradiction_count >= 3 AND pinned = 0
                       AND NOT EXISTS (
                         SELECT 1 FROM fact_conflicts c
                         WHERE c.status = 'open'
                           AND (c.fact_a_id = facts.fact_id OR c.fact_b_id = facts.fact_id)
                       )""",
                ),
            ]
            for reason, predicate in rules:
                rows = self._conn.execute(
                    f"SELECT fact_id FROM facts WHERE status != 'archived' AND {predicate}"
                ).fetchall()
                for row in rows:
                    if self.archive_fact(int(row["fact_id"]), reason=reason):
                        archived.append(int(row["fact_id"]))
            return {"archived": archived, "count": len(archived)}

    def set_hippocampus_paused(self, paused: bool) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO hippocampus_control(key, value, updated_at)
                VALUES ('paused', ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE
                SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
                """,
                ("true" if paused else "false",),
            )

    def hippocampus_status(self) -> dict:
        with self._lock:
            paused = self._conn.execute(
                "SELECT value FROM hippocampus_control WHERE key = 'paused'"
            ).fetchone()
            queued = self._conn.execute(
                "SELECT status, COUNT(*) AS count FROM hippocampus_sessions GROUP BY status"
            ).fetchall()
            last = self._conn.execute(
                "SELECT * FROM hippocampus_runs ORDER BY run_id DESC LIMIT 1"
            ).fetchone()
            return {
                "paused": bool(paused and paused["value"] == "true"),
                "queue": {row["status"]: row["count"] for row in queued},
                "last_run": dict(last) if last else None,
            }

    def hippocampus_history(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM hippocampus_runs ORDER BY run_id DESC LIMIT ?",
                (max(1, min(100, int(limit))),),
            ).fetchall()
            return [dict(row) for row in rows]

    def record_conflict(self, fact_a_id: int, fact_b_id: int, reason: str = "") -> int:
        """Persist a conflict without overwriting either memory."""
        if fact_a_id == fact_b_id:
            raise ValueError("a memory cannot conflict with itself")
        a, b = sorted((int(fact_a_id), int(fact_b_id)))
        with self._lock:
            found = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE fact_id IN (?, ?)", (a, b)
            ).fetchone()[0]
            if found != 2:
                raise KeyError("one or both conflict facts do not exist")
            inserted = self._conn.execute(
                """
                INSERT OR IGNORE INTO fact_conflicts (fact_a_id, fact_b_id, reason)
                VALUES (?, ?, ?)
                """,
                (a, b, reason),
            )
            row = self._conn.execute(
                "SELECT conflict_id FROM fact_conflicts WHERE fact_a_id = ? AND fact_b_id = ?",
                (a, b),
            ).fetchone()
            self._conn.execute(
                "UPDATE facts SET status = 'conflicted', updated_at = CURRENT_TIMESTAMP WHERE fact_id IN (?, ?)",
                (a, b),
            )
            if inserted.rowcount:
                self.record_evidence(
                    a,
                    "contradict",
                    provenance_type="agent",
                    detail=f"Conflicts with memory {b}: {reason}",
                    weight=0.5,
                )
                self.record_evidence(
                    b,
                    "contradict",
                    provenance_type="agent",
                    detail=f"Conflicts with memory {a}: {reason}",
                    weight=0.5,
                )
            return int(row["conflict_id"])

    def list_conflicts(self, status: str = "open", limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT c.*, a.content AS fact_a_content, b.content AS fact_b_content
                FROM fact_conflicts c
                JOIN facts a ON a.fact_id = c.fact_a_id
                JOIN facts b ON b.fact_id = c.fact_b_id
                WHERE c.status = ? ORDER BY c.detected_at DESC LIMIT ?
                """,
                (status, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_facts(
        self,
        category: str | None = None,
        min_trust: float = 0.0,
        limit: int = 50,
        include_archived: bool = False,
    ) -> list[dict]:
        """Browse facts ordered by trust_score descending.

        Optionally filter by category and minimum trust score.
        """
        with self._lock:
            params: list = [min_trust]
            status_clause = "" if include_archived else "AND status != 'archived'"
            category_clause = ""
            if category is not None:
                category_clause = "AND category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at,
                       memory_kind, status, provenance_type, provenance_ref,
                       provenance_json, confirmation_count, contradiction_count,
                       last_confirmed_at, archived_at
                FROM facts
                WHERE trust_score >= ?
                  {status_clause}
                  {category_clause}
                ORDER BY trust_score DESC
                LIMIT ?
            """
            rows = self._conn.execute(sql, params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def record_feedback(self, fact_id: int, helpful: bool) -> dict:
        """Record user feedback and adjust trust asymmetrically.

        helpful=True  -> trust += 0.05, helpful_count += 1
        helpful=False -> trust -= 0.10

        Returns a dict with fact_id, old_trust, new_trust, helpful_count.
        Raises KeyError if fact_id does not exist.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score, helpful_count FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"fact_id {fact_id} not found")

            old_trust: float = row["trust_score"]
            delta = _HELPFUL_DELTA if helpful else _UNHELPFUL_DELTA
            new_trust = _clamp_trust(old_trust + delta)

            helpful_increment = 1 if helpful else 0
            self._conn.execute(
                """
                UPDATE facts
                SET trust_score    = ?,
                    helpful_count  = helpful_count + ?,
                    updated_at     = CURRENT_TIMESTAMP
                WHERE fact_id = ?
                """,
                (new_trust, helpful_increment, fact_id),
            )
            self._conn.commit()

            self._conn.execute(
                """
                INSERT INTO fact_evidence
                    (fact_id, polarity, source_type, detail, weight)
                VALUES (?, ?, 'user', 'explicit feedback', 1.0)
                """,
                (fact_id, "confirm" if helpful else "contradict"),
            )
            return {
                "fact_id": fact_id,
                "old_trust": old_trust,
                "new_trust": new_trust,
                "helpful_count": row["helpful_count"] + helpful_increment,
            }

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entity candidates from text using simple regex rules.

        Rules applied (in order):
        1. Capitalized multi-word phrases  e.g. "John Doe"
        2. Double-quoted terms             e.g. "Python"
        3. Single-quoted terms             e.g. 'pytest'
        4. AKA patterns                    e.g. "Guido aka BDFL" -> two entities

        Returns a deduplicated list preserving first-seen order.
        """
        seen: set[str] = set()
        candidates: list[str] = []

        def _add(name: str) -> None:
            stripped = name.strip()
            if stripped and stripped.lower() not in seen:
                seen.add(stripped.lower())
                candidates.append(stripped)

        for m in _RE_CAPITALIZED.finditer(text):
            _add(m.group(1))

        for m in _RE_DOUBLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_SINGLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_AKA.finditer(text):
            _add(m.group(1))
            _add(m.group(2))

        return candidates

    def _resolve_entity(self, name: str) -> int:
        """Find an existing entity by name or alias (case-insensitive) or create one.

        Returns the entity_id.
        """
        # Exact name match
        row = self._conn.execute(
            "SELECT entity_id FROM entities WHERE name LIKE ?", (name,)
        ).fetchone()
        if row is not None:
            return int(row["entity_id"])

        # Search aliases — aliases stored as comma-separated; use LIKE with % boundaries
        alias_row = self._conn.execute(
            """
            SELECT entity_id FROM entities
            WHERE ',' || aliases || ',' LIKE '%,' || ? || ',%'
            """,
            (name,),
        ).fetchone()
        if alias_row is not None:
            return int(alias_row["entity_id"])

        # Create new entity
        cur = self._conn.execute("INSERT INTO entities (name) VALUES (?)", (name,))
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[return-value]

    def _link_fact_entity(self, fact_id: int, entity_id: int) -> None:
        """Insert into fact_entities, silently ignore if the link already exists."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO fact_entities (fact_id, entity_id)
            VALUES (?, ?)
            """,
            (fact_id, entity_id),
        )
        self._conn.commit()

    def _compute_hrr_vector(self, fact_id: int, content: str) -> None:
        """Compute and store HRR vector for a fact. No-op if numpy unavailable."""
        with self._lock:
            if not self._hrr_available:
                return

            # Get entities linked to this fact
            rows = self._conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fact_id,),
            ).fetchall()
            entities = [row["name"] for row in rows]

            vector = hrr.encode_fact(content, entities, self.hrr_dim)
            self._conn.execute(
                "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
                (hrr.phases_to_bytes(vector), fact_id),
            )
            self._conn.commit()

    def _rebuild_bank(self, category: str) -> None:
        """Full rebuild of a category's memory bank from all its fact vectors."""
        with self._lock:
            if not self._hrr_available:
                return

            bank_name = f"cat:{category}"
            rows = self._conn.execute(
                "SELECT hrr_vector FROM facts WHERE category = ? AND status != 'archived' AND hrr_vector IS NOT NULL",
                (category,),
            ).fetchall()

            if not rows:
                self._conn.execute(
                    "DELETE FROM memory_banks WHERE bank_name = ?", (bank_name,)
                )
                self._conn.commit()
                return

            vectors = [hrr.bytes_to_phases(row["hrr_vector"]) for row in rows]
            bank_vector = hrr.bundle(*vectors)
            fact_count = len(vectors)

            # Check SNR
            hrr.snr_estimate(self.hrr_dim, fact_count)

            self._conn.execute(
                """
                INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(bank_name) DO UPDATE SET
                    vector = excluded.vector,
                    dim = excluded.dim,
                    fact_count = excluded.fact_count,
                    updated_at = excluded.updated_at
                """,
                (bank_name, hrr.phases_to_bytes(bank_vector), self.hrr_dim, fact_count),
            )
            self._conn.commit()

    def rebuild_all_vectors(self, dim: int | None = None) -> int:
        """Recompute all HRR vectors + banks from text. For recovery/migration.

        Returns the number of facts processed.
        """
        with self._lock:
            if not self._hrr_available:
                return 0

            if dim is not None:
                self.hrr_dim = dim

            rows = self._conn.execute(
                "SELECT fact_id, content, category FROM facts"
            ).fetchall()

            categories: set[str] = set()
            for row in rows:
                self._compute_hrr_vector(row["fact_id"], row["content"])
                categories.add(row["category"])

            for category in categories:
                self._rebuild_bank(category)

            return len(rows)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict."""
        result = dict(row)
        if "trust_score" in result:
            result["confidence"] = result["trust_score"]
        if "provenance_json" in result:
            try:
                result["provenance"] = json.loads(result.pop("provenance_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                result["provenance"] = {}
        return result

    def close(self) -> None:
        """Release this instance's reference to the shared connection.

        The underlying connection is closed only when the last MemoryStore
        referencing the same database is closed, so closing one instance can
        never break sibling instances that still hold it. Idempotent.
        """
        if getattr(self, "_entry", None) is None:
            return
        with MemoryStore._shared_guard:
            entry = self._entry
            if entry is None:
                return
            entry["refs"] -= 1
            if entry["refs"] <= 0:
                try:
                    entry["conn"].close()
                finally:
                    MemoryStore._shared.pop(self._key, None)
            self._entry = None

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
