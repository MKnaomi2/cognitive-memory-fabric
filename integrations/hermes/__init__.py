"""Thin Hermes tool adapter for the standalone Cognitive Memory Fabric."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO = Path(
    os.environ.get(
        "COGNITIVE_MEMORY_FABRIC_REPO",
        os.environ.get(
            "HIPPOCAMPAL_MEMORY_REPO",
            r"C:\Hermes\cognitive-memory-fabric",
        ),
    )
).resolve()
_SOURCE = _REPO / "src"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from hippocampal_memory.coordination import MemoryCoordinator  # noqa: E402
from hippocampal_memory.cognition import CognitiveMemorySystem  # noqa: E402
from hippocampal_memory.replay import HippocampusEngine  # noqa: E402
from hippocampal_memory.vault import VaultSynchronizer  # noqa: E402

from hippocampal_memory.adapters.hermes import create_engine  # noqa: E402
from hippocampal_memory.provider import CognitiveMemoryProvider, ProviderConfig  # noqa: E402

_engine: HippocampusEngine | None = None
_provider: CognitiveMemoryProvider | None = None


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
        context_id=str(args.get("context_id") or ""),
        event_start_at=str(args.get("event_start_at") or "") or None,
        event_end_at=str(args.get("event_end_at") or "") or None,
        autobiographical=bool(args.get("autobiographical", False)),
        self_relevance=float(args.get("self_relevance", 0.0)),
        perspective=str(args.get("perspective") or "unknown"),
        recollection_mode=str(args.get("recollection_mode") or "know"),
        vividness=float(args.get("vividness", 0.0)),
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
        if _provider is not None:
            rows = _provider.readout.search(query)["memories"][:limit]
        else:
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


def _context(args: dict, **_: Any) -> str:
    cognition = CognitiveMemorySystem(_get_engine().store)
    action = str(args.get("action") or "reinstate")
    if action == "reinstate":
        result = cognition.reinstate_context(
            cue=str(args.get("cue") or ""),
            memory_id=(
                int(args["memory_id"]) if args.get("memory_id") is not None else None
            ),
            limit=int(args.get("limit") or 20),
        )
    elif action == "temporal_order":
        result = {
            "memories": cognition.temporal_order(
                context_id=str(args.get("context_id") or "") or None,
                event_id=str(args.get("event_id") or "") or None,
            )
        }
    elif action == "recency":
        result = {
            "memories": cognition.recall_recent(
                limit=int(args.get("limit") or 20),
                half_life_hours=float(args.get("half_life_hours") or 168),
                context_id=str(args.get("context_id") or "") or None,
            )
        }
    elif action == "autobiography":
        result = {"memories": cognition.autobiographical_timeline(args.get("limit", 100))}
    else:
        return _result({"ok": False, "error": "unsupported context action"})
    return _result({"ok": True, **result})


def _reactivate(args: dict, **_: Any) -> str:
    result = CognitiveMemorySystem(_get_engine().store).reactivate(
        int(args["memory_id"]),
        cue=str(args.get("cue") or ""),
        prediction_error=float(args.get("prediction_error") or 0.0),
        retrieval_duration_seconds=float(
            args.get("retrieval_duration_seconds") or 0.0
        ),
    )
    return _result({"ok": True, **result})


def _reconsolidate(args: dict, **context: Any) -> str:
    result = CognitiveMemorySystem(_get_engine().store).reconsolidate(
        str(args["window_id"]),
        polarity=str(args["polarity"]),
        provenance_type=str(args.get("provenance_type") or "user"),
        provenance_ref=str(
            args.get("source_ref") or context.get("session_id") or ""
        ),
        detail=str(args.get("detail") or ""),
        weight=float(args.get("weight") or 1.0),
        contextual_updates=dict(args.get("contextual_updates") or {}),
    )
    return _result({"ok": True, **result})


def _cognitive_status(args: dict, **_: Any) -> str:
    cognition = CognitiveMemorySystem(_get_engine().store)
    memory_id = args.get("memory_id")
    result = cognition.status()
    if memory_id is not None:
        result["source_monitoring"] = cognition.monitor_source(int(memory_id))
    return _result({"ok": True, **result})


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
            ,"context_id": {"type": "string"}
            ,"event_start_at": {"type": "string"}
            ,"event_end_at": {"type": "string"}
            ,"autobiographical": {"type": "boolean"}
            ,"self_relevance": {"type": "number", "minimum": 0, "maximum": 1}
            ,"perspective": {"type": "string", "enum": ["field", "observer", "semantic", "unknown"]}
            ,"recollection_mode": {"type": "string", "enum": ["remember", "know", "inferred", "unknown"]}
            ,"vividness": {"type": "number", "minimum": 0, "maximum": 1}
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
CONTEXT_SCHEMA = {
    "name": "hippocampal_context",
    "description": "Reinstate temporal context, recall explicit order/recency, or inspect autobiographical memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["reinstate", "temporal_order", "recency", "autobiography"]},
            "memory_id": {"type": "integer"},
            "cue": {"type": "string"},
            "context_id": {"type": "string"},
            "event_id": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            "half_life_hours": {"type": "number", "minimum": 0.01}
        },
        "required": ["action"]
    }
}
REACTIVATE_SCHEMA = {
    "name": "hippocampal_reactivate",
    "description": "Reinstate a memory and, only when boundary conditions pass, open a time-bounded reconsolidation window.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "integer"},
            "cue": {"type": "string"},
            "prediction_error": {"type": "number", "minimum": 0, "maximum": 1},
            "retrieval_duration_seconds": {"type": "number", "minimum": 0}
        },
        "required": ["memory_id", "cue"]
    }
}
RECONSOLIDATE_SCHEMA = {
    "name": "hippocampal_reconsolidate",
    "description": "Integrate cited evidence into a labile memory while preserving before/after versions.",
    "parameters": {
        "type": "object",
        "properties": {
            "window_id": {"type": "string"},
            "polarity": {"type": "string", "enum": ["confirm", "contradict"]},
            "provenance_type": {"type": "string", "enum": ["user", "agent", "web", "reflection", "sensor", "system"]},
            "source_ref": {"type": "string"},
            "detail": {"type": "string"},
            "weight": {"type": "number", "minimum": 0, "maximum": 1},
            "contextual_updates": {"type": "object"}
        },
        "required": ["window_id", "polarity", "detail"]
    }
}
COGNITIVE_STATUS_SCHEMA = {
    "name": "hippocampal_cognitive_status",
    "description": "Inspect temporal contexts, events, source monitoring, reconsolidation, and operational self-recollection state.",
    "parameters": {
        "type": "object",
        "properties": {"memory_id": {"type": "integer"}}
    }
}


def _check() -> tuple[bool, str]:
    available = _SOURCE.is_dir()
    return (
        available,
        "" if available else f"standalone source not found at {_SOURCE}",
    )


def register(ctx) -> None:
    global _provider
    engine = _get_engine()
    values = {}
    try:
        import yaml

        config_path = engine.store.db_path.parent / "config.yaml"
        root = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
        memory = root.get("memory", {})
        if memory.get("provider") == "cognitive-memory-fabric":
            values = memory
    except Exception:
        values = {}
    provider = CognitiveMemoryProvider(
        engine.store,
        ProviderConfig(
            replay_mode=str(values.get("replay_mode", "none")),
            candidate_limit=int(values.get("candidate_limit", 50)),
            recall_limit=int(values.get("recall_limit", 10)),
            max_injected_chars=int(values.get("max_injected_chars", 8000)),
            deadline_seconds=float(values.get("deadline_seconds", 2.0)),
        ),
        circuit=_load_circuit(engine, values),
        tool_schemas=[
            schema
            for _, schema, _, _ in _TOOLS
        ],
        tool_handlers={
            name: handler
            for name, _, handler, _ in _TOOLS
        },
    )
    _provider = provider
    register_provider = getattr(ctx, "register_memory_provider", None)
    if callable(register_provider):
        register_provider(provider)
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="hippocampal_memory",
            schema=schema,
            handler=handler,
            check_fn=_check,
            emoji=emoji,
        )


def _load_circuit(engine, values):
    if str(values.get("replay_mode", "none")) != "neural":
        return None
    try:
        from hippocampal_memory.circuit import TrisynapticCircuit

        row = engine.store._conn.execute(
            """
            SELECT path FROM neural_checkpoints
            WHERE circuit_version='trisynaptic-v3-content-readout'
            ORDER BY created_at DESC LIMIT 1
            """
        ).fetchone()
        if row:
            return TrisynapticCircuit.from_checkpoint(
                row["path"], device=str(values.get("device", "cuda"))
            )
    except Exception:
        return None
    return None


_TOOLS = (
    ("hippocampal_remember", REMEMBER_SCHEMA, _remember, "🧠"),
    ("hippocampal_query", QUERY_SCHEMA, _query, "🔎"),
    ("hippocampal_evidence", EVIDENCE_SCHEMA, _evidence, "⚖️"),
    ("hippocampal_archive", ARCHIVE_SCHEMA, _archive, "🗄️"),
    ("hippocampal_vault_sync", VAULT_SCHEMA, _vault_sync, "🔗"),
    ("hippocampal_context", CONTEXT_SCHEMA, _context, "🕰️"),
    ("hippocampal_reactivate", REACTIVATE_SCHEMA, _reactivate, "♻️"),
    (
        "hippocampal_reconsolidate",
        RECONSOLIDATE_SCHEMA,
        _reconsolidate,
        "🧬",
    ),
    (
        "hippocampal_cognitive_status",
        COGNITIVE_STATUS_SCHEMA,
        _cognitive_status,
        "🧭",
    ),
)
