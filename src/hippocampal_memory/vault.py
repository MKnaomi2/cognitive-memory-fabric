"""Concept-centric Obsidian projection with journaled, reversible writes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .coordination import MemoryCoordinator
from .cognition import CognitiveMemorySystem
from .store import MemoryStore

MANAGED_START = "<!-- hippocampal:managed:start -->"
MANAGED_END = "<!-- hippocampal:managed:end -->"
_UNSAFE = re.compile(r"[^a-z0-9]+")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_-]{2,}")
_STOP_WORDS = {
    "and",
    "are",
    "for",
    "from",
    "has",
    "have",
    "into",
    "its",
    "that",
    "the",
    "their",
    "this",
    "was",
    "were",
    "with",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _slug(value: str) -> str:
    return (_UNSAFE.sub("-", value.lower()).strip("-") or "memory")[:64].rstrip("-")


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse the flat, deterministic frontmatter emitted by this package."""
    text = text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        return {}, text
    boundary = text.find("\n---\n", 4)
    if boundary < 0:
        return {}, text
    fields: dict[str, Any] = {}
    for line in text[4:boundary].splitlines():
        if ":" not in line or line.startswith((" ", "\t")):
            continue
        key, raw = line.split(":", 1)
        try:
            fields[key.strip()] = json.loads(raw.strip())
        except json.JSONDecodeError:
            fields[key.strip()] = raw.strip()
    return fields, text[boundary + 5 :]


@dataclass(frozen=True)
class VaultMutation:
    operation: str
    memory_id: str
    relative_path: str
    before_sha256: str
    after_sha256: str
    content: str
    reason: str


@dataclass(frozen=True)
class NeuralNoteLink:
    """One bounded, explainable neural association projected into the vault."""

    memory_id: str
    note_path: str
    title: str
    score: float
    neural_overlap: float
    reasons: tuple[str, ...]


def _tokens(content: str) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(content.lower())
        if token not in _STOP_WORDS
    }


def _signature(raw: Any) -> set[int]:
    try:
        values = json.loads(str(raw or "[]"))
        return {int(value) for value in values}
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()


class VaultProjector:
    """Render authoritative memory state without overwriting human notes."""

    FOLDERS = {
        "episode": "Memories/Episodes",
        "fact": "Neocortex/Concepts",
        "principle": "Neocortex/Principles",
        "identity": "Neocortex/Identity",
    }

    def note_path(self, memory: dict[str, Any]) -> str:
        folder = (
            "Archive"
            if memory.get("status") == "archived"
            else self.FOLDERS.get(
                str(memory.get("memory_kind", "fact")), "Neocortex/Concepts"
            )
        )
        return (
            f"{folder}/{_slug(str(memory['content']))}"
            f"--m{int(memory['fact_id']):06d}.md"
        )

    def render(
        self,
        memory: dict[str, Any],
        *,
        note_id: str,
        revision: int,
        existing_text: str = "",
        related_notes: Iterable[NeuralNoteLink] = (),
    ) -> str:
        existing_meta, existing_body = parse_frontmatter(existing_text)
        user_body = self._human_body(existing_body)
        try:
            provenance = json.loads(str(memory.get("provenance_json", "{}")))
        except json.JSONDecodeError:
            provenance = {}
        content = str(memory["content"]).strip()
        fields = {
            "id": note_id,
            "memory_id": str(memory["fact_id"]),
            "title": content.splitlines()[0][:120],
            "memory_kind": memory.get("memory_kind", "fact"),
            "status": memory.get("status", "active"),
            "confidence": round(float(memory.get("trust_score", 0.5)), 6),
            "evidence_count": int(memory.get("evidence_count", 0)),
            "provenance_type": memory.get("provenance_type", "imported"),
            "source_ref": memory.get("provenance_ref", ""),
            "source_uri": provenance.get("source_uri", ""),
            "valid_from": memory.get("valid_from"),
            "valid_until": memory.get("valid_until"),
            "context_id": memory.get("context_id") or "",
            "event_id": memory.get("event_id") or "",
            "sequence_index": memory.get("sequence_index"),
            "event_start_at": memory.get("event_start_at"),
            "event_end_at": memory.get("event_end_at"),
            "temporal_uncertainty_seconds": memory.get(
                "temporal_uncertainty_seconds", 0
            ),
            "autobiographical": bool(memory.get("autobiographical", False)),
            "self_relevance": round(float(memory.get("self_relevance", 0)), 6),
            "perspective": memory.get("perspective", "unknown"),
            "recollection_mode": memory.get("recollection_mode", "know"),
            "vividness": round(float(memory.get("vividness", 0)), 6),
            "source_memory_score": round(
                float(memory.get("source_memory_score") or 0), 6
            ),
            "superseded_by": memory.get("superseded_by"),
            "engram_id": memory.get("engram_id") or "",
            "time_cell_count": int(memory.get("time_cell_count") or 0),
            "consolidation_state": (
                "consolidated"
                if memory.get("memory_kind") in {"principle", "identity"}
                else "episodic"
            ),
            "sync_revision": revision,
            "updated": memory.get("updated_at") or _now(),
        }
        for key in ("aliases", "tags"):
            if key in existing_meta:
                fields[key] = existing_meta[key]
        frontmatter = "\n".join(
            f"{key}: {_scalar(value)}" for key, value in fields.items()
        )
        source = (
            f"{memory.get('provenance_type', 'imported')}: "
            f"{memory.get('provenance_ref') or 'unspecified source'}"
        )
        relation = (
            f"- Superseded by memory `{memory['superseded_by']}`"
            if memory.get("superseded_by")
            else "- None recorded"
        )
        neural_links = list(related_notes)
        if neural_links:
            rendered_links = []
            for link in neural_links:
                target = link.note_path.removesuffix(".md")
                title = link.title.replace("[", "").replace("]", "").replace("|", " ")
                reasons = "; ".join(link.reasons)
                rendered_links.append(
                    f"- [[{target}|{title}]] — association "
                    f"{link.score:.3f}; neural overlap "
                    f"{link.neural_overlap:.3f}; {reasons}"
                )
            relationship_text = relation + "\n\n### Neural associations\n\n" + "\n".join(
                rendered_links
            )
        else:
            relationship_text = relation + "\n\n### Neural associations\n\n- None yet"
        managed = (
            f"{MANAGED_START}\n# {fields['title']}\n\n{content}\n\n"
            f"## Provenance\n\n- {source}\n\n"
            f"## Confidence\n\n"
            f"- Score: {float(memory.get('trust_score', 0.5)):.3f}\n"
            f"- Confirmations: {int(memory.get('confirmation_count', 0))}\n"
            f"- Contradictions: {int(memory.get('contradiction_count', 0))}\n\n"
            f"## Temporal context\n\n"
            f"- Context: {memory.get('context_id') or 'unbound'}\n"
            f"- Event: {memory.get('event_id') or 'unsegmented'}\n"
            f"- Sequence: {memory.get('sequence_index')}\n"
            f"- Interval: {memory.get('event_start_at')} → "
            f"{memory.get('event_end_at')}\n\n"
            f"## Recollection\n\n"
            f"- Autobiographical: {bool(memory.get('autobiographical', False))}\n"
            f"- Mode: {memory.get('recollection_mode', 'know')}\n"
            f"- Perspective: {memory.get('perspective', 'unknown')}\n"
            f"- Vividness: {float(memory.get('vividness', 0)):.3f}\n"
            f"- Source-monitoring score: "
            f"{float(memory.get('source_memory_score') or 0):.3f}\n\n"
            f"## Relationships\n\n{relationship_text}\n{MANAGED_END}\n"
        )
        result = f"---\n{frontmatter}\n---\n\n{managed}"
        if user_body:
            result += f"\n\n## Human notes\n\n{user_body}\n"
        return result

    @staticmethod
    def _human_body(body: str) -> str:
        if MANAGED_START not in body or MANAGED_END not in body:
            return body.strip()
        tail = body.split(MANAGED_END, 1)[1]
        if "## Human notes" in tail:
            tail = tail.split("## Human notes", 1)[1]
        return tail.strip()


class VaultSynchronizer:
    """Plan and apply bounded vault mutations with rollback journals."""

    def __init__(
        self,
        store: MemoryStore,
        vault_root: str | Path,
        coordinator: MemoryCoordinator | None = None,
    ) -> None:
        self.store = store
        self.root = Path(vault_root).expanduser().resolve()
        self.coordinator = coordinator or MemoryCoordinator(store)
        self.cognition = CognitiveMemorySystem(store, self.coordinator)
        self.projector = VaultProjector()

    def plan(self, memory_ids: Iterable[int] | None = None) -> list[VaultMutation]:
        with self.store._lock:
            where, params = "", []
            if memory_ids is not None:
                params = [int(value) for value in memory_ids]
                if not params:
                    return []
                where = "WHERE f.fact_id IN (" + ",".join("?" for _ in params) + ")"
            selected_rows = self.store._conn.execute(
                f"""
                SELECT f.*,
                    (SELECT COUNT(*) FROM fact_evidence e
                     WHERE e.fact_id=f.fact_id) evidence_count,
                    b.engram_id, b.ca1_signature_json,
                    (SELECT source_memory_score
                     FROM source_monitoring_assessments s
                     WHERE s.fact_id=f.fact_id
                     ORDER BY assessed_at DESC LIMIT 1) source_memory_score,
                    (SELECT json_array_length(cell_ids_json)
                     FROM time_cell_bindings t
                     WHERE t.memory_id=CAST(f.fact_id AS TEXT)) time_cell_count
                FROM facts f LEFT JOIN engram_bindings b
                  ON b.memory_id=CAST(f.fact_id AS TEXT)
                {where} ORDER BY f.fact_id
                """,
                params,
            ).fetchall()
            relationship_rows = self.store._conn.execute(
                """
                SELECT f.fact_id,f.content,f.status,f.memory_kind,
                       f.context_id,f.event_id,
                       b.ca1_signature_json
                FROM facts f JOIN engram_bindings b
                  ON b.memory_id=CAST(f.fact_id AS TEXT)
                WHERE f.status='active'
                  AND b.encoding_version='content-v3'
                  AND COALESCE(b.ca1_signature_json,'[]')!='[]'
                ORDER BY f.fact_id
                """
            ).fetchall()
        related = self._neural_links(relationship_rows)
        mutations = []
        for row in selected_rows:
            memory = dict(row)
            memory_id = str(memory["fact_id"])
            relative = self.projector.note_path(memory)
            target = self._target(relative)
            existing_bytes = target.read_bytes() if target.exists() else b""
            existing = existing_bytes.decode("utf-8") if existing_bytes else ""
            note_id = self.coordinator.register_vault_note(memory_id, relative)
            revision = self.coordinator.current_revision(f"memory:{memory_id}")
            content = self.projector.render(
                memory,
                note_id=note_id,
                revision=revision,
                existing_text=existing,
                related_notes=related.get(memory_id, ()),
            )
            before = _hash(existing_bytes) if existing_bytes else ""
            after = _hash(content.encode())
            if before != after:
                mutations.append(
                    VaultMutation(
                        "update" if existing else "create",
                        memory_id,
                        relative,
                        before,
                        after,
                        content,
                        "project authoritative lifecycle state",
                    )
                )
        return mutations

    def _neural_links(
        self,
        rows: Iterable[Any],
        *,
        max_links: int = 5,
        minimum_neural_overlap: float = 0.10,
        minimum_score: float = 0.18,
    ) -> dict[str, list[NeuralNoteLink]]:
        """Derive bounded vault links from neural overlap plus corroborating context."""
        memories: dict[str, dict[str, Any]] = {}
        by_neuron: dict[int, list[str]] = {}
        for row in rows:
            memory = dict(row)
            memory_id = str(memory["fact_id"])
            signature = _signature(memory.get("ca1_signature_json"))
            if not signature:
                continue
            memory["signature"] = signature
            memory["tokens"] = _tokens(str(memory.get("content") or ""))
            memory["note_path"] = self.projector.note_path(memory)
            memories[memory_id] = memory
            for neuron_id in signature:
                by_neuron.setdefault(neuron_id, []).append(memory_id)
        token_frequency: dict[str, int] = {}
        for memory in memories.values():
            for token in memory["tokens"]:
                token_frequency[token] = token_frequency.get(token, 0) + 1
        distinctive_frequency_limit = max(3, (len(memories) + 19) // 20)

        intersections: dict[str, dict[str, int]] = {
            memory_id: {} for memory_id in memories
        }
        for memory_ids in by_neuron.values():
            for index, left in enumerate(memory_ids):
                for right in memory_ids[index + 1 :]:
                    intersections[left][right] = intersections[left].get(right, 0) + 1
                    intersections[right][left] = intersections[right].get(left, 0) + 1

        links: dict[str, list[NeuralNoteLink]] = {}
        for memory_id, candidates in intersections.items():
            source = memories[memory_id]
            scored: list[NeuralNoteLink] = []
            for candidate_id, shared_neurons in candidates.items():
                target = memories[candidate_id]
                neural_overlap = shared_neurons / min(
                    len(source["signature"]), len(target["signature"])
                )
                if neural_overlap < minimum_neural_overlap:
                    continue
                shared_tokens = source["tokens"] & target["tokens"]
                distinctive_tokens = {
                    token
                    for token in shared_tokens
                    if token_frequency.get(token, 0) <= distinctive_frequency_limit
                }
                union = source["tokens"] | target["tokens"]
                lexical_overlap = len(shared_tokens) / len(union) if union else 0.0
                same_event = bool(
                    source.get("event_id")
                    and source.get("event_id") == target.get("event_id")
                )
                same_context = bool(
                    source.get("context_id")
                    and source.get("context_id") == target.get("context_id")
                )
                corroborated = (
                    same_event
                    or same_context
                    or (bool(distinctive_tokens) and lexical_overlap >= 0.12)
                )
                if not corroborated:
                    continue
                score = min(
                    1.0,
                    0.70 * neural_overlap
                    + 0.15 * lexical_overlap
                    + (0.10 if same_context else 0.0)
                    + (0.15 if same_event else 0.0),
                )
                if score < minimum_score:
                    continue
                reasons = []
                if same_event:
                    reasons.append("same event")
                if same_context:
                    reasons.append("same context")
                if distinctive_tokens:
                    terms = ", ".join(sorted(distinctive_tokens)[:4])
                    reasons.append(f"shared terms: {terms}")
                scored.append(
                    NeuralNoteLink(
                        memory_id=candidate_id,
                        note_path=str(target["note_path"]),
                        title=str(target["content"]).splitlines()[0][:120],
                        score=score,
                        neural_overlap=neural_overlap,
                        reasons=tuple(reasons),
                    )
                )
            scored.sort(key=lambda item: (-item.score, int(item.memory_id)))
            links[memory_id] = scored[: max(1, min(12, int(max_links)))]
        return links

    def apply(
        self,
        mutations: Iterable[VaultMutation],
        *,
        max_mutations: int = 25,
        journal_root: str | Path | None = None,
    ) -> dict[str, Any]:
        selected = list(mutations)
        if len(selected) > max_mutations:
            raise ValueError(
                f"refusing {len(selected)} vault writes; limit is {max_mutations}"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        base = (
            Path(journal_root).expanduser().resolve()
            if journal_root
            else self.root / ".hippocampal-memory" / "journals"
        )
        journal = base / (
            datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        )
        journal.mkdir(parents=True)
        applied: list[tuple[VaultMutation, Path, Path | None]] = []
        manifest = []
        try:
            for change in selected:
                target, backup = self._target(change.relative_path), None
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if _hash(target.read_bytes()) != change.before_sha256:
                        raise RuntimeError(
                            f"concurrent vault edit: {change.relative_path}"
                        )
                    backup = journal / "before" / change.relative_path
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, backup)
                descriptor, name = tempfile.mkstemp(
                    prefix=".hippocampal-", suffix=".tmp", dir=target.parent
                )
                os.close(descriptor)
                temporary = Path(name)
                try:
                    temporary.write_bytes(change.content.encode("utf-8"))
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
                applied.append((change, target, backup))
                revision = self.coordinator.current_revision(
                    f"memory:{change.memory_id}"
                )
                self.coordinator.record_sync(
                    change.memory_id,
                    direction="memory_to_vault",
                    note_path=change.relative_path,
                    from_revision=max(0, revision - 1),
                    to_revision=revision,
                    before_sha256=change.before_sha256,
                    after_sha256=change.after_sha256,
                    outcome="applied",
                )
                manifest.append(
                    {
                        "operation": change.operation,
                        "memory_id": change.memory_id,
                        "relative_path": change.relative_path,
                        "before_sha256": change.before_sha256,
                        "after_sha256": change.after_sha256,
                        "backup": (
                            str(backup.relative_to(journal)) if backup else None
                        ),
                    }
                )
            (journal / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at": _now(),
                        "vault_root": str(self.root),
                        "mutations": manifest,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return {"status": "applied", "count": len(applied), "journal": str(journal)}
        except Exception:
            for change, target, backup in reversed(applied):
                if backup and backup.exists():
                    shutil.copy2(backup, target)
                elif change.operation == "create":
                    target.unlink(missing_ok=True)
            raise

    def _target(self, relative: str) -> Path:
        target = (self.root / relative).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("vault path traversal rejected") from exc
        return target
