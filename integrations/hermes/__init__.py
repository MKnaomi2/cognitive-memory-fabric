"""Thin Hermes tool adapter for the standalone hippocampal-memory package."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO = Path(
    os.environ.get("HIPPOCAMPAL_MEMORY_REPO", r"C:\Hermes\hippocampal-memory")
).resolve()
_SOURCE = _REPO / "src"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from hippocampal_memory.coordination import MemoryCoordinator  # noqa: E402
from hippocampal_memory.replay import HippocampusEngine  # noqa: E402
from hippocampal_memory.vault import VaultSynchronizer  # noqa: E402

from hippocampal_memory.adapters.hermes import create_engine  # noqa: E402

_engine: HippocampusEngine | None = None


def _get_engine() -> HippocampusEngine:
    global _engine
    if _engine is None:
        _engine = create_engine()
        MemoryCoordinator(_engine.store)
    return _engine


def _result(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)


def _remember(args: dict, **context: Any) -> str:
    content = str(args.get("content") or "").strip()
    if not content:
        return _result({"ok": False, "error": "content is required"})
    kind = str(args.get("memory_kind") or "fact")
    actor_type = str(args.get("provenance_type") or "user")
    session_id = str(context.get("session_id") or args.get("source_ref") or "")
    coordinator = MemoryCoordinator(_get_engine().store)
    memory_id = coordinator.ingest(
        content,
        actor_type=actor_type,
        actor_ref=session_id,
        source_uri=str(args.get("source_uri") or ""),
        memory_kind=kind,
        confidence=float(args.get("confidence", 0.6)),
        subject_key=str(args.get("subject_key") or ""),
        predicate_key=str(args.get("predicate_key") or ""),
        pinned=bool(args.get("pinned", False)),
    )
    return _result({"ok": True, "memory_id": memory_id, "revision": coordinator.current_revision(f"memory:{memory_id}")})


def _query(args: dict, **_: Any) -> str:
    engine = _get_engine()
    query = str(args.get("query") or "").strip()
    limit = min(50, max(1, int(args.get("limit") or 10)))
    include_archived = bool(args.get("include_archived", False))
    if query and include_archived:
        candidates = engine.store.list_facts(limit=200, include_archived=True)
        terms = query.casefold().split()
        rows = [
            row
            for row in candidates
            if all(term in str(row["content"]).casefold() for term in terms)
        ][:limit]
    elif query:
        rows = engine.store.search_facts(query, limit=limit)
    else:
        rows = engine.store.list_facts(
            limit=limit, include_archived=include_archived
        )
    return _result({"ok": True, "memories": rows})


def _evidence(args: dict, **context: Any) -> str:
    engine = _get_engine()
    result = engine.store.record_evidence(
        int(args["memory_id"]),
        str(args["polarity"]),
        provenance_type=str(args.get("provenance_type") or "user"),
        provenance_ref=str(
            args.get("source_ref") or context.get("session_id") or ""
        ),
        detail=str(args.get("detail") or ""),
        weight=float(args.get("weight", 1.0)),
    )
    coordinator = MemoryCoordinator(engine.store)
    event = coordinator.append_event(
        f"memory:{int(args['memory_id'])}",
        f"evidence.{args['polarity']}",
        {"detail": str(args.get("detail") or ""), "weight": float(args.get("weight", 1.0))},
        actor_type=str(args.get("provenance_type") or "user"),
        actor_ref=str(args.get("source_ref") or context.get("session_id") or ""),
    )
    return _result({"ok": True, "memory": result, "revision": event.revision})


def _archive(args: dict, **context: Any) -> str:
    memory_id = int(args["memory_id"])
    reason = str(args.get("reason") or "explicit Hermes request")
    engine = _get_engine()
    archived = engine.store.archive_fact(memory_id, reason)
    if archived:
        MemoryCoordinator(engine.store).append_event(
            f"memory:{memory_id}",
            "memory.archived",
            {"reason": reason},
            actor_type="user",
            actor_ref=str(context.get("session_id") or ""),
        )
    return _result({"ok": archived, "memory_id": memory_id, "archived": archived})


def _vault_sync(args: dict, **_: Any) -> str:
    engine = _get_engine()
    vault = Path(str(args.get("vault") or r"C:\Hermes\Knowledge"))
    synchronizer = VaultSynchronizer(
        engine.store, vault, MemoryCoordinator(engine.store)
    )
    plan = synchronizer.plan()
    apply = bool(args.get("apply", False))
    limit = min(25, max(1, int(args.get("limit") or 25)))
    if not apply:
        return _result(
            {
                "ok": True,
                "mode": "dry-run",
                "count": len(plan),
                "paths": [item.relative_path for item in plan[:limit]],
            }
        )
    result = synchronizer.apply(plan[:limit], max_mutations=limit)
    result.update(ok=True, remaining=max(0, len(plan) - limit))
    return _result(result)


REMEMBER_SCHEMA = {
    "name": "hippocampal_remember",
    "description": "Store a durable memory with explicit provenance and confidence.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "memory_kind": {"type": "string", "enum": ["episode", "fact", "principle", "identity"]},
            "provenance_type": {"type": "string", "enum": ["user", "agent", "web", "reflection", "sensor", "system"]},
            "source_ref": {"type": "string"},
            "source_uri": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "subject_key": {"type": "string"},
            "predicate_key": {"type": "string"},
            "pinned": {"type": "boolean"}
        },
        "required": ["content"]
    }
}
QUERY_SCHEMA = {
    "name": "hippocampal_query",
    "description": "Search provenance-aware active or archived memories.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            "include_archived": {"type": "boolean"}
        }
    }
}
EVIDENCE_SCHEMA = {
    "name": "hippocampal_evidence",
    "description": "Confirm or contradict a memory with cited evidence.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "integer"},
            "polarity": {"type": "string", "enum": ["confirm", "contradict"]},
            "provenance_type": {"type": "string", "enum": ["user", "agent", "web", "reflection", "sensor", "system"]},
            "source_ref": {"type": "string"},
            "detail": {"type": "string"},
            "weight": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["memory_id", "polarity"]
    }
}
ARCHIVE_SCHEMA = {
    "name": "hippocampal_archive",
    "description": "Reversibly archive a no-longer-relevant memory; never hard-deletes.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "integer"},
            "reason": {"type": "string"}
        },
        "required": ["memory_id"]
    }
}
VAULT_SCHEMA = {
    "name": "hippocampal_vault_sync",
    "description": "Plan or apply a journaled Obsidian projection, at most 25 notes per call.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault": {"type": "string"},
            "apply": {"type": "boolean"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 25}
        }
    }
}


def _check() -> tuple[bool, str]:
    available = _SOURCE.is_dir()
    return (
        available,
        "" if available else f"standalone source not found at {_SOURCE}",
    )


def register(ctx) -> None:
    for name, schema, handler, emoji in (
        ("hippocampal_remember", REMEMBER_SCHEMA, _remember, "🧠"),
        ("hippocampal_query", QUERY_SCHEMA, _query, "🔎"),
        ("hippocampal_evidence", EVIDENCE_SCHEMA, _evidence, "⚖️"),
        ("hippocampal_archive", ARCHIVE_SCHEMA, _archive, "🗄️"),
        ("hippocampal_vault_sync", VAULT_SCHEMA, _vault_sync, "🔗"),
    ):
        ctx.register_tool(
            name=name,
            toolset="hippocampal_memory",
            schema=schema,
            handler=handler,
            check_fn=_check,
            emoji=emoji,
        )
