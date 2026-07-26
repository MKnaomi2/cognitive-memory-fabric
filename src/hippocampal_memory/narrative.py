"""Evidence-grounded narrative composition and governed consolidation."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from .readout import MemoryReadout, ReadoutConfig
from .store import MemoryStore

ALGORITHM_VERSION = "narrative-v1"
_WORDS = re.compile(r"[a-z0-9][a-z0-9_-]{2,}")
_STRUCTURES = {"adaptive", "chronological", "thematic", "problem-decision-outcome"}
_RATINGS = {"helpful", "unhelpful", "missing"}


@dataclass(frozen=True)
class NarrativeClaim:
    text: str
    kind: str
    confidence: float
    source_ids: tuple[int, ...]
    relation: str = ""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _signature(raw: Any) -> set[int]:
    try:
        return {int(value) for value in json.loads(str(raw or "[]"))}
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()


class NarrativeEngine:
    """Connect retrieved memories without promoting association to fact."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self._neural_candidates: list[tuple[dict[str, Any], set[int]]] | None = None
        self._recall_cache: dict[str, dict[str, Any]] = {}

    def compose(
        self,
        query: str,
        *,
        max_memories: int = 12,
        structure: str = "adaptive",
        include_neural: bool = True,
    ) -> dict[str, Any]:
        query = str(query or "").strip()
        if not query:
            raise ValueError("query is required")
        if structure not in _STRUCTURES:
            raise ValueError(f"invalid narrative structure: {structure}")
        limit = max(3, min(20, int(max_memories)))
        result = self._recall_cache.get(query)
        if result is None:
            result = MemoryReadout(
                self.store,
                ReadoutConfig(mode="symbolic", candidate_limit=50, recall_limit=6),
            ).search(query)
            self._recall_cache[query] = result
        memories = {int(row["fact_id"]): dict(row) for row in result["memories"]}
        seed_ids = list(memories)
        self._expand_explicit(memories, seed_ids, limit)
        self._expand_temporal(memories, seed_ids, limit)
        if include_neural:
            self._expand_neural(memories, seed_ids, limit)
        ordered = self._order(list(memories.values()), structure, query)
        selected_structure = self._select_structure(ordered, structure, query)
        claims = self._claims(ordered, include_neural=include_neural)
        return {
            "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "structure": selected_structure,
            "memory_count": len(ordered),
            "memories": [self._public_memory(row) for row in ordered],
            "claims": [
                {
                    "text": claim.text,
                    "kind": claim.kind,
                    "confidence": claim.confidence,
                    "source_ids": list(claim.source_ids),
                    "relation": claim.relation,
                }
                for claim in claims
            ],
            "story": self._render(claims, selected_structure),
            "fallback": bool(result.get("fallback")),
        }

    def _fetch(self, fact_id: int) -> dict[str, Any] | None:
        row = self.store._conn.execute(
            """
            SELECT f.*,b.ca1_signature_json FROM facts f
            LEFT JOIN engram_bindings b ON b.memory_id=CAST(f.fact_id AS TEXT)
            WHERE f.fact_id=? AND f.status!='archived'
            """,
            (int(fact_id),),
        ).fetchone()
        return dict(row) if row else None

    def _expand_explicit(
        self, memories: dict[int, dict[str, Any]], seed_ids: list[int], limit: int
    ) -> None:
        for fact_id in seed_ids:
            if len(memories) >= limit:
                return
            rows = self.store._conn.execute(
                """
                SELECT source_fact_id related FROM fact_derivations
                WHERE derived_fact_id=?
                UNION SELECT derived_fact_id FROM fact_derivations
                WHERE source_fact_id=?
                UNION SELECT fact_b_id FROM fact_conflicts WHERE fact_a_id=?
                UNION SELECT fact_a_id FROM fact_conflicts WHERE fact_b_id=?
                """,
                (fact_id, fact_id, fact_id, fact_id),
            ).fetchall()
            for row in rows:
                related = self._fetch(int(row["related"]))
                if related:
                    memories.setdefault(int(related["fact_id"]), related)
                if len(memories) >= limit:
                    return

    def _expand_temporal(
        self, memories: dict[int, dict[str, Any]], seed_ids: list[int], limit: int
    ) -> None:
        for fact_id in seed_ids:
            if len(memories) >= limit:
                return
            seed = memories[fact_id]
            if not seed.get("event_id") and not seed.get("context_id"):
                continue
            rows = self.store._conn.execute(
                """
                SELECT f.*,b.ca1_signature_json FROM facts f
                LEFT JOIN engram_bindings b
                  ON b.memory_id=CAST(f.fact_id AS TEXT)
                WHERE f.status!='archived'
                  AND (
                    (? != '' AND f.event_id=?)
                    OR (? != '' AND f.context_id=?)
                  )
                ORDER BY ABS(COALESCE(f.sequence_index,0)-?)
                LIMIT 8
                """,
                (
                    seed.get("event_id") or "",
                    seed.get("event_id") or "",
                    seed.get("context_id") or "",
                    seed.get("context_id") or "",
                    int(seed.get("sequence_index") or 0),
                ),
            ).fetchall()
            for row in rows:
                memories.setdefault(int(row["fact_id"]), dict(row))
                if len(memories) >= limit:
                    return

    def _expand_neural(
        self, memories: dict[int, dict[str, Any]], seed_ids: list[int], limit: int
    ) -> None:
        if len(memories) >= limit:
            return
        if self._neural_candidates is None:
            rows = self.store._conn.execute(
                """
                SELECT f.*,b.ca1_signature_json FROM facts f JOIN engram_bindings b
                  ON b.memory_id=CAST(f.fact_id AS TEXT)
                WHERE f.status='active' AND b.encoding_version='content-v3'
                  AND COALESCE(b.ca1_signature_json,'[]')!='[]'
                """
            ).fetchall()
            self._neural_candidates = [
                (dict(row), _signature(row["ca1_signature_json"])) for row in rows
            ]
        candidates = [
            (row, signature)
            for row, signature in self._neural_candidates
            if int(row["fact_id"]) not in memories
        ]
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for seed_id in seed_ids:
            source = _signature(memories[seed_id].get("ca1_signature_json"))
            if not source:
                continue
            for candidate, target in candidates:
                if not target:
                    continue
                overlap = len(source & target) / max(1, min(len(source), len(target)))
                if overlap >= 0.10:
                    scored.append((overlap, int(candidate["fact_id"]), candidate))
        for _, fact_id, row in sorted(scored, key=lambda item: (-item[0], item[1])):
            memories.setdefault(fact_id, row)
            if len(memories) >= limit:
                return

    @staticmethod
    def _select_structure(
        rows: list[dict[str, Any]], requested: str, query: str
    ) -> str:
        if requested != "adaptive":
            return requested
        lowered = query.lower()
        if any(word in lowered for word in ("why", "decision", "outcome", "problem")):
            return "problem-decision-outcome"
        if sum(row.get("sequence_index") is not None for row in rows) >= 2:
            return "chronological"
        return "thematic"

    @staticmethod
    def _order(
        rows: list[dict[str, Any]], requested: str, query: str
    ) -> list[dict[str, Any]]:
        selected = NarrativeEngine._select_structure(rows, requested, query)
        if selected == "chronological":
            return sorted(
                rows,
                key=lambda row: (
                    str(row.get("event_start_at") or row.get("created_at") or ""),
                    int(row.get("sequence_index") or 0),
                    int(row["fact_id"]),
                ),
            )
        return rows

    def _claims(
        self, rows: list[dict[str, Any]], *, include_neural: bool
    ) -> list[NarrativeClaim]:
        claims = [
            NarrativeClaim(
                str(row["content"]).strip(),
                "remembered",
                round(float(row.get("trust_score") or 0), 6),
                (int(row["fact_id"]),),
            )
            for row in rows
        ]
        compared_pairs: set[tuple[int, int]] = set()
        for left, right in zip(rows, rows[1:]):
            pair = tuple(sorted((int(left["fact_id"]), int(right["fact_id"]))))
            compared_pairs.add(pair)
            relation = self._relation(left, right, include_neural=include_neural)
            if relation:
                claims.append(
                    NarrativeClaim(
                        relation[1],
                        relation[0],
                        relation[2],
                        (int(left["fact_id"]), int(right["fact_id"])),
                        relation[3],
                    )
                )
        if include_neural and not any(
            claim.relation == "neural_association" for claim in claims
        ):
            association_added = False
            for index, left in enumerate(rows):
                for right in rows[index + 1 :]:
                    pair = tuple(
                        sorted((int(left["fact_id"]), int(right["fact_id"])))
                    )
                    if pair in compared_pairs:
                        continue
                    relation = self._relation(
                        left, right, neural_only=True, include_neural=True
                    )
                    if relation:
                        claims.append(
                            NarrativeClaim(
                                relation[1],
                                relation[0],
                                relation[2],
                                (int(left["fact_id"]), int(right["fact_id"])),
                                relation[3],
                            )
                        )
                        association_added = True
                        break
                if association_added:
                    break
        return claims

    def _relation(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        *,
        neural_only: bool = False,
        include_neural: bool = True,
    ) -> tuple[str, str, float, str] | None:
        left_id, right_id = int(left["fact_id"]), int(right["fact_id"])
        if not neural_only:
            conflict = self.store._conn.execute(
                """
                SELECT 1 FROM fact_conflicts
                WHERE status='open' AND (
                  (fact_a_id=? AND fact_b_id=?) OR (fact_a_id=? AND fact_b_id=?)
                )
                """,
                (left_id, right_id, right_id, left_id),
            ).fetchone()
            if conflict:
                return (
                    "remembered",
                    "The sources disagree; both versions must remain visible.",
                    1.0,
                    "conflicts",
                )
            if left.get("event_id") and left.get("event_id") == right.get("event_id"):
                return (
                    "remembered",
                    "These memories belong to the same recorded event sequence.",
                    0.95,
                    "same_event",
                )
            if left.get("context_id") and left.get("context_id") == right.get("context_id"):
                return (
                    "inference",
                    "These memories may form one thread because they share context.",
                    0.65,
                    "same_context",
                )
        if not include_neural:
            return None
        left_sig = _signature(left.get("ca1_signature_json"))
        right_sig = _signature(right.get("ca1_signature_json"))
        overlap = len(left_sig & right_sig) / max(1, min(len(left_sig), len(right_sig)))
        if overlap >= 0.10:
            return (
                "inference",
                "The neural index associates these memories, but does not establish causation.",
                round(overlap, 6),
                "neural_association",
            )
        return None

    @staticmethod
    def _render(claims: list[NarrativeClaim], structure: str) -> str:
        lines = [f"## Evidence-grounded narrative ({structure})"]
        for claim in claims:
            label = {
                "remembered": "Remembered",
                "inference": "Inference",
                "uncertain": "Uncertain",
            }[claim.kind]
            citations = " ".join(f"[m{fact_id}]" for fact_id in claim.source_ids)
            lines.append(f"- **{label}:** {claim.text} {citations}")
        if not claims:
            lines.append("- **Uncertain:** No supporting memories were retrieved.")
        return "\n".join(lines)

    @staticmethod
    def _public_memory(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "memory_id": int(row["fact_id"]),
            "content": str(row["content"]),
            "confidence": float(row.get("trust_score") or 0),
            "provenance_type": str(row.get("provenance_type") or "unknown"),
            "provenance_ref": str(row.get("provenance_ref") or ""),
            "context_id": str(row.get("context_id") or ""),
            "event_id": str(row.get("event_id") or ""),
            "sequence_index": row.get("sequence_index"),
        }

    def consolidate_drafts(
        self, sleep_session_id: str, *, max_threads: int = 4
    ) -> dict[str, Any]:
        """Create bounded drafts from well-supported contexts after neural sleep."""
        stale = self.refresh_stale()
        groups = self.store._conn.execute(
            """
            SELECT context_id,COUNT(*) memory_count,AVG(trust_score) confidence
            FROM facts WHERE status='active' AND context_id!=''
            GROUP BY context_id HAVING COUNT(*)>=3 AND AVG(trust_score)>=0.70
            ORDER BY MAX(updated_at) DESC LIMIT ?
            """,
            (max(1, min(12, int(max_threads))),),
        ).fetchall()
        created = updated = promoted = 0
        for group in groups:
            context_id = str(group["context_id"])
            rows = self.store._conn.execute(
                """
                SELECT * FROM facts WHERE status='active' AND context_id=?
                ORDER BY event_start_at,sequence_index,fact_id LIMIT 12
                """,
                (context_id,),
            ).fetchall()
            source_ids = [int(row["fact_id"]) for row in rows]
            if self._has_open_conflict(source_ids):
                continue
            thread_key = "context:" + context_id
            existing = self.store._conn.execute(
                "SELECT * FROM narrative_threads WHERE thread_key=?", (thread_key,)
            ).fetchone()
            thread_id = str(existing["thread_id"]) if existing else str(uuid.uuid4())
            passes = int(existing["support_passes"]) + 1 if existing else 1
            status = "active" if passes >= 2 else "draft"
            version_id = str(uuid.uuid4())
            summary = " ".join(
                str(row["content"]).strip().splitlines()[0] for row in rows[:6]
            )[:4000]
            fingerprint = hashlib.sha256(_json(source_ids).encode()).hexdigest()
            if existing:
                self.store._conn.execute(
                    """
                    UPDATE narrative_threads SET support_passes=?,status=?,
                      current_version_id=?,updated_at=CURRENT_TIMESTAMP
                    WHERE thread_id=?
                    """,
                    (passes, status, version_id, thread_id),
                )
                updated += 1
            else:
                self.store._conn.execute(
                    """
                    INSERT INTO narrative_threads(
                      thread_id,thread_key,title,structure,status,
                      current_version_id,support_passes
                    ) VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        thread_id,
                        thread_key,
                        f"Context narrative: {context_id}"[:160],
                        "chronological",
                        status,
                        version_id,
                        passes,
                    ),
                )
                created += 1
            self.store._conn.execute(
                """
                INSERT INTO narrative_versions(
                  version_id,thread_id,summary,confidence,model_digest,
                  algorithm_version,source_fingerprint
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    version_id,
                    thread_id,
                    summary,
                    float(group["confidence"]),
                    "",
                    ALGORITHM_VERSION,
                    fingerprint,
                ),
            )
            for position, row in enumerate(rows):
                claim_id = str(uuid.uuid4())
                self.store._conn.execute(
                    """
                    INSERT INTO narrative_claims VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        claim_id,
                        version_id,
                        position,
                        str(row["content"]),
                        "remembered",
                        float(row["trust_score"]),
                        "context_member",
                    ),
                )
                self.store._conn.execute(
                    "INSERT INTO narrative_sources VALUES (?,?,?,?)",
                    (claim_id, int(row["fact_id"]), 1.0, "supports"),
                )
            if status == "active" and (not existing or existing["status"] != "active"):
                promoted += 1
        self.store._conn.commit()
        return {
            "status": "completed",
            "sleep_session_id": sleep_session_id,
            "created": created,
            "updated": updated,
            "promoted": promoted,
            "stale": stale,
        }

    def refresh_stale(self) -> int:
        rows = self.store._conn.execute(
            """
            SELECT DISTINCT t.thread_id FROM narrative_threads t
            JOIN narrative_claims nc ON nc.version_id=t.current_version_id
            JOIN narrative_sources ns ON ns.claim_id=nc.claim_id
            JOIN facts f ON f.fact_id=ns.fact_id
            WHERE t.status='active' AND (
              f.status!='active' OR EXISTS (
                SELECT 1 FROM fact_conflicts c WHERE c.status='open'
                  AND (c.fact_a_id=f.fact_id OR c.fact_b_id=f.fact_id)
              )
            )
            """
        ).fetchall()
        for row in rows:
            self.store._conn.execute(
                """
                UPDATE narrative_threads SET status='stale',
                  updated_at=CURRENT_TIMESTAMP WHERE thread_id=?
                """,
                (str(row["thread_id"]),),
            )
        if rows:
            self.store._conn.commit()
        return len(rows)

    def _has_open_conflict(self, source_ids: list[int]) -> bool:
        if not source_ids:
            return False
        marks = ",".join("?" for _ in source_ids)
        return bool(
            self.store._conn.execute(
                f"""
                SELECT 1 FROM fact_conflicts WHERE status='open'
                  AND (fact_a_id IN ({marks}) OR fact_b_id IN ({marks}))
                LIMIT 1
                """,
                (*source_ids, *source_ids),
            ).fetchone()
        )

    def feedback(
        self,
        *,
        rating: str,
        thread_id: str | None = None,
        audit_id: int | None = None,
        detail: str = "",
    ) -> dict[str, Any]:
        if rating not in _RATINGS:
            raise ValueError("rating must be helpful, unhelpful, or missing")
        if not thread_id and audit_id is None:
            raise ValueError("thread_id or audit_id is required")
        feedback_id = str(uuid.uuid4())
        self.store._conn.execute(
            "INSERT INTO narrative_feedback VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)",
            (feedback_id, thread_id, audit_id, rating, detail[:500]),
        )
        promoted = False
        if thread_id:
            column = "helpful_count" if rating == "helpful" else "unhelpful_count"
            self.store._conn.execute(
                f"""
                UPDATE narrative_threads SET {column}={column}+1,
                  updated_at=CURRENT_TIMESTAMP WHERE thread_id=?
                """,
                (thread_id,),
            )
            row = self.store._conn.execute(
                "SELECT * FROM narrative_threads WHERE thread_id=?", (thread_id,)
            ).fetchone()
            if row and rating == "helpful" and int(row["support_passes"]) >= 1:
                self.store._conn.execute(
                    "UPDATE narrative_threads SET status='active' WHERE thread_id=?",
                    (thread_id,),
                )
                promoted = True
        self.store._conn.commit()
        return {
            "feedback_id": feedback_id,
            "rating": rating,
            "thread_id": thread_id,
            "audit_id": audit_id,
            "promoted": promoted,
        }

    def list_threads(self, *, status: str = "active", limit: int = 20) -> list[dict]:
        if status not in {"draft", "active", "stale", "all"}:
            raise ValueError("invalid narrative status")
        clause, params = ("", []) if status == "all" else ("WHERE t.status=?", [status])
        rows = self.store._conn.execute(
            f"""
            SELECT t.*,v.summary,v.confidence,v.algorithm_version
            FROM narrative_threads t LEFT JOIN narrative_versions v
              ON v.version_id=t.current_version_id
            {clause} ORDER BY t.updated_at DESC LIMIT ?
            """,
            (*params, max(1, min(100, int(limit)))),
        ).fetchall()
        return [dict(row) for row in rows]
