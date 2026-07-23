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
from .store import MemoryStore

MANAGED_START = "<!-- hippocampal:managed:start -->"
MANAGED_END = "<!-- hippocampal:managed:end -->"
_UNSAFE = re.compile(r"[^a-z0-9]+")


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
            "superseded_by": memory.get("superseded_by"),
            "engram_id": memory.get("engram_id") or "",
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
        managed = (
            f"{MANAGED_START}\n# {fields['title']}\n\n{content}\n\n"
            f"## Provenance\n\n- {source}\n\n"
            f"## Confidence\n\n"
            f"- Score: {float(memory.get('trust_score', 0.5)):.3f}\n"
            f"- Confirmations: {int(memory.get('confirmation_count', 0))}\n"
            f"- Contradictions: {int(memory.get('contradiction_count', 0))}\n\n"
            f"## Relationships\n\n{relation}\n{MANAGED_END}\n"
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
        self.projector = VaultProjector()

    def plan(self, memory_ids: Iterable[int] | None = None) -> list[VaultMutation]:
        with self.store._lock:
            where, params = "", []
            if memory_ids is not None:
                params = [int(value) for value in memory_ids]
                if not params:
                    return []
                where = "WHERE f.fact_id IN (" + ",".join("?" for _ in params) + ")"
            rows = self.store._conn.execute(
                f"""
                SELECT f.*,
                    (SELECT COUNT(*) FROM fact_evidence e
                     WHERE e.fact_id=f.fact_id) evidence_count,
                    b.engram_id
                FROM facts f LEFT JOIN engram_bindings b
                  ON b.memory_id=CAST(f.fact_id AS TEXT)
                {where} ORDER BY f.fact_id
                """,
                params,
            ).fetchall()
        mutations = []
        for row in rows:
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
