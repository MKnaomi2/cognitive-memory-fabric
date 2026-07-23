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
- immutable versioned events, optimistic revisions, and synchronization ledgers
- concept-centric Obsidian projection that preserves human annotations
- an actual 36,864-neuron EC→DG→CA3→CA1 circuit with 770,048 sparse synapses
- local STDP, homeostasis, inhibition, NREM ripple/spindle replay, and REM replay
- a loopback-only WebGPU/WebGL neural observatory with live and recorded views

## Architecture

SQLite is the hippocampus and source of lifecycle truth. It stores episodes,
facts, evidence, confidence, conflicts, event revisions, engram bindings,
checkpoints, and vault synchronization history.

Obsidian is the neocortical projection. It stores consolidated, human-readable
concept, principle, identity, conflict, archive, and map notes. Managed sections
are regenerated from SQLite; text under `## Human notes` is preserved.

The neural circuit is a third, local state projection. Language models may
choose a replay candidate, but they cannot submit synaptic weights. Weight
changes come only from spike timing and homeostatic rules.

Every cross-layer write is attributable:

`memory event → SQLite revision → engram binding → vault sync ledger`

## Neural extras

For the RTX 5060 Ti install the CUDA 13 PyTorch wheel in an isolated
environment, then install the observatory:

```powershell
python -m venv .venv-neural
.\.venv-neural\Scripts\python -m pip install `
  "torch==2.12.1+cu130" --index-url https://download.pytorch.org/whl/cu130
.\.venv-neural\Scripts\python -m pip install -e ".[observatory,obsidian,dev]"
```

The circuit acceptance probe is:

```powershell
.\.venv-neural\Scripts\hippocampal-memory circuit-check --device cuda
```

Run the API and observatory UI locally:

```powershell
.\.venv-neural\Scripts\hippocampal-memory observatory
cd viewer
npm run dev
```

Open `http://localhost:3000`. The viewer uses WebGPU when available and WebGL2
otherwise. It renders every neuron, switches to region clusters at distant
zoom, shows active synapses, supports orbit/pan/fly controls, and can scrub
local `.hmrec` sleep recordings. It cannot change memory lifecycle state.

To start both loopback services automatically at Windows logon:

```powershell
.\scripts\Install-NeuralObservatoryTask.ps1 -Enable
```

## Obsidian

Dry-run a projection first:

```powershell
hippocampal-memory vault-plan --vault C:\Hermes\Knowledge
```

Apply at most 25 journaled writes:

```powershell
hippocampal-memory vault-sync --vault C:\Hermes\Knowledge --limit 25 --apply
```

`VaultMigrator.stage()` repairs a complete copy of a vault: invalid metadata,
duplicate identifiers, exact duplicates, unresolved or ambiguous links,
generic README names, canonical maps, semantic relationship metadata, and
local-only routing. `cutover()` refuses any staging vault that does not pass
acceptance and preserves the previous vault as an archive.

The optional desktop adapter in `obsidian-plugin/` opens the same observatory in
an Obsidian tab without granting it note-write access.

## GPU sleep

`SleepConsolidator` acquires an exclusive GPU window, asks loopback Ollama to
release its model, and preempts within one second when foreground work begins.
It performs bounded encoding, NREM, and REM phases; writes a hashed checkpoint
and length-prefixed MessagePack recording; and restores normal on-demand GPU
use when it exits.

The Windows task installer creates `Hermes_Hippocampal_Sleep` at 2:10 AM with
idle, battery, and single-instance guards:

```powershell
.\scripts\Install-HippocampalSleepTask.ps1 -Enable
```

The legacy replay task should remain disabled after cutover.

## Hermes tools

The adapter in `integrations/hermes/` registers five native tools:

- `hippocampal_remember`
- `hippocampal_query`
- `hippocampal_evidence`
- `hippocampal_archive`
- `hippocampal_vault_sync`

This lets Hermes update the system when explicitly asked while keeping
provenance, evidence, bounded vault writes, and archive-before-delete policy.

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
