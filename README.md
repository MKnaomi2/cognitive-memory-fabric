# Hippocampal Memory

A standalone Python package for provenance-aware memory, evidence-driven
confidence, conflict detection, systems consolidation, identity meta-memory,
and reversible forgetting.

It is independent of the Hermes repository. A small optional adapter connects
it to Hermes, while the lifecycle store and replay worker can be embedded in
other agents.

## What it provides

- source type, source reference, evidence history, and temporal validity
- confidence that rises with confirmation and falls with contradiction
- explicit conflicts and same-property supersession
- archive-before-delete forgetting with restoration
- conservative consolidation into reusable principles
- stricter multi-source, multi-day gates for identity memories
- checkpointed local-model replay with grounded message citations
- secret-shaped input redaction, no replay tools, and bounded context
- idle/GPU deferral, foreground preemption, pause/resume, and decision audit

## Install

```bash
pip install -e .
```

NumPy is optional and enables holographic vector retrieval:

```bash
pip install -e ".[holographic]"
```

## Minimal use

```python
from hippocampal_memory import MemoryStore

store = MemoryStore("memory.db")
fact_id = store.add_fact(
    "Large refactors benefit from hierarchical planning.",
    memory_kind="principle",
    provenance_type="reflection",
    provenance_ref="run:42",
    confidence=0.8,
)
store.record_evidence(
    fact_id,
    polarity="confirm",
    source_type="session",
    source_ref="session:abc",
)
```

## Replay

The included SQLite transcript source expects:

- `sessions(id, started_at, ended_at)`
- `messages(id, session_id, role, content, tool_name, timestamp, active)`

This is the schema used by the Hermes adapter, but it can be produced by any
host agent. Replay sends only user/assistant text to a loopback Ollama endpoint.
Tool payloads and hidden reasoning are excluded.

```bash
hippocampal-memory --home ./data --state-db ./state.db \
  --model hermes-local:latest run --mode backfill --shadow
```

Remove `--shadow` after reviewing the result. Other commands are `status`,
`pause`, `resume`, `history`, `digest`, and `maintain`.

## Evidence gates

Principles require at least three supporting observations, two independent
sources, confidence of 0.70, and no open conflicts.

Identity memories require at least five observations, three independent
sources, confidence of 0.80, seven days of evidence, and no open conflicts.

Model output is always a proposal. Deterministic policy validates citations and
evidence thresholds before a memory is accepted.

## Forgetting

Forgetting archives rather than deletes. Pinned memories are protected.
Consolidated source episodes receive a seven-day grace period, and restored
memories receive a 30-day review exemption.

## Hermes adapter

Install the optional YAML dependency, then:

```python
from hippocampal_memory.adapters.hermes import create_engine

engine = create_engine()
result = engine.run("micro")
```

The host can wrap foreground turns with
`hippocampal_memory.activity.foreground_turn()` so maintenance yields
immediately.

## Privacy

This repository contains code and tests only. Runtime memories, transcripts,
SQLite databases, configuration, logs, credentials, and model files are
excluded by `.gitignore`.

## Origin

The initial lifecycle and replay implementation was extracted from an
MIT-licensed Hermes Agent integration. The core package is maintained
independently; the original Nous Research copyright is retained in `LICENSE`.
