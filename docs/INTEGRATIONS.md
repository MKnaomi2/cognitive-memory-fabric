# Integrations

## Obsidian as neocortical projection

SQLite remains authoritative. Obsidian provides durable human-readable concepts,
maps, annotations, and archives.

### Projection layout

| Memory state/kind | Default folder |
|---|---|
| Episode | `Memories/Episodes` |
| Fact | `Neocortex/Concepts` |
| Principle | `Neocortex/Principles` |
| Identity | `Neocortex/Identity` |
| Promoted narrative | `Narratives/Active` |
| Archived | `Archive` |

Note filenames combine a normalized content slug with a stable memory suffix,
for example `large-refactors-benefit--m000042.md`.

### Managed frontmatter

Projected notes include stable note/memory IDs, title, kind, status, confidence,
evidence count, provenance type/reference/URI, validity interval,
supersession, engram ID, consolidation state, sync revision, and update time.
Existing aliases and tags are preserved.

The generated body is fenced by:

```text
<!-- hippocampal:managed:start -->
...
<!-- hippocampal:managed:end -->
```

Text under `## Human notes` is retained across regeneration. A legacy note
without managed markers is treated as human-authored content and retained.
Narrative drafts remain in SQLite. Only promoted, non-stale narratives are
newly projected, with links to every supporting memory note. If a projected
narrative later becomes stale, its existing note is visibly marked stale and
must be revalidated before use.

### Synchronization transaction

1. `vault-plan` renders desired content without writing.
2. Stable registry identity maps each memory to exactly one note/path.
3. Plans include before/after SHA-256 values.
4. `vault-sync --apply` refuses more than the configured bound (default 25).
5. The synchronizer verifies the note still matches its planned before hash.
6. Existing content is backed up in a per-run journal.
7. New content is atomically replaced from a same-directory temporary file.
8. A sync-ledger row stores revisions, paths, hashes, and outcome.
9. A failure reverses already-applied writes in reverse order.

All target paths are resolved and containment-checked beneath the vault root.

## Full-vault migration

`VaultMigrator` separates repair from cutover.

Vault projection can also turn durable neural associations into managed
Obsidian links. Each memory note receives at most five related-note links.
Links require CA1 signature overlap plus shared context, event, or meaningful
content terms, unless the neural overlap is independently strong. The managed
relationship entry includes the combined score, neural overlap, and its
corroborating reasons; human-authored note content remains untouched.

### Staging

`stage()` copies the source vault and repairs the copy. It can:

- normalize required frontmatter and source metadata;
- produce deterministic unique IDs;
- classify memory kind and consolidation state;
- normalize sensitivities/status/source types/timestamps;
- repair malformed paths and generic README names;
- create canonical maps and Home connectivity;
- add semantic relationship metadata;
- archive exact duplicate notes;
- resolve ambiguous links conservatively;
- create quarantined recovered-link stubs for missing targets; and
- flag local-only policy violations.

### Acceptance

Audit checks include:

- required canonical directories and maps;
- valid frontmatter and primary maps;
- duplicate IDs/content;
- malformed paths and generic README nodes;
- unresolved links;
- semantic links missing metadata;
- orphans and Home connectivity;
- local-only violations; and
- graph degree/connectivity.

### Cutover

`cutover()` refuses a staging vault that does not pass acceptance. A successful
cutover renames the previous live vault to the explicitly selected archive path
and promotes the validated staging directory. The source vault is never edited
in place during staging.

## Obsidian desktop adapter

`obsidian-plugin/` registers a Hermes Neural Observatory view and command. It
embeds `http://localhost:3000` in an Obsidian tab. The plugin does not receive
the lifecycle publisher token and does not write notes.

Install its files beneath:

```text
<vault>\.obsidian\plugins\hermes-neural-observatory\
```

and enable `hermes-neural-observatory` in the vault's community plugin list.

## Hermes Agent adapter

`integrations/hermes/` is deliberately thin. It imports the standalone package,
registers a first-class `CognitiveMemoryProvider`, and retains eleven tools.

The provider participates in Hermes initialization, system-prompt policy,
bounded prefetch, explicit durable-turn synchronization, and shutdown. Prefetch
uses at most 50 candidates, injects at most 10 memories or 8,000 characters,
and has a two-second deadline. Returned content is wrapped as untrusted evidence
with provenance, confidence, validity, conflict, and supersession state.
Recalled context and tool output are not automatically written back.

Neural mode uses the frozen v0.5.1 candidate defaults: lexical cues, weight
`0.05`, margin `0.0`, and activation threshold `0.70`. These values are written
explicitly during provider installation and may be overridden in the Hermes
`memory` configuration.

Hermes itself does not need Torch. A persistent authenticated service runs in
the isolated CUDA environment, loads the latest hash-verified checkpoint, and
reloads when sleep registers a newer checkpoint:

```powershell
.\scripts\Run-NeuralReadout.ps1
```

The service binds only to `127.0.0.1:8767`. Its bearer token is generated under
the Hermes runtime directory and is never stored in `config.yaml`. Configure
Hermes with:

```yaml
memory:
  provider: cognitive-memory-fabric
  replay_mode: neural
  neural_service_url: http://127.0.0.1:8767
  cue_mode: lexical
  neural_weight: 0.05
  neural_margin_min: 0.0
  neural_activation_min: 0.7
  neural_shadow: true
  neural_rollout_percent: 0
  capture_turns: false
  turn_capture_max_chars: 6000
```

If the service, token, CUDA device, or checkpoint is unavailable, the provider
fails closed to symbolic replay rather than failing the Hermes turn.
With `neural_shadow: true`, Hermes executes the real neural readout but returns
the symbolic order. It stores only query hashes, candidate orders, latency,
checkpoint identity, applied weight, and fallback status in
`neural_readout_audit`; raw queries are not retained. Once the narrative
evaluation and explicit feedback
gates pass, `neural_shadow` may be disabled and `neural_rollout_percent` raised
gradually. Query hashing assigns a stable rollout bucket, so the same query does
not switch arms unpredictably.

`hippocampal_narrative` composes a bounded story from retrieved memories,
temporal/event neighbors, explicit derivations or conflicts, and qualified
neural associations. Every claim carries source memory IDs and is visibly
labeled remembered, inference, or uncertain. `hippocampal_narrative_feedback`
records only an explicit helpful, unhelpful, or missing rating.

Completed-turn capture is local-only and opt-in. With `capture_turns: true`,
the provider stores one bounded `episode` containing the cleaned user request
and final assistant response. It does not persist tool transcripts, subagent or
cron turns, trivial acknowledgements, detected credentials, or unbounded
content. Captured episodes enter the same provenance, evidence, neural sleep,
and archive lifecycle as explicit memories. The default remains `false` so a
fresh installation cannot silently begin retaining ordinary conversation.

Provider installation is dry-run by default and backs up `config.yaml` before
an applied change:

```powershell
cognitive-memory hermes install
cognitive-memory hermes install --apply
cognitive-memory hermes doctor
cognitive-memory hermes uninstall --apply
```

The explicit tool surface is:

The `hippocampal_*` identifiers below are retained compatibility interfaces
from the project's former name. They operate on the Cognitive Memory Fabric;
the name does not imply that the lifecycle, evidence, vault, or temporal
cognition features belong to the neural replay subsystem.

| Tool | Operation |
|---|---|
| `hippocampal_remember` | create/reobserve a memory with type, provenance, confidence, and optional subject/predicate/validity data |
| `hippocampal_query` | search active memories or explicitly include archives |
| `hippocampal_evidence` | record weighted source/confirm/contradict evidence and append an event |
| `hippocampal_archive` | reversibly archive a memory with an attributed reason |
| `hippocampal_vault_sync` | dry-run or apply a bounded Obsidian projection |
| `hippocampal_context` | reinstate context or retrieve explicit order, recency, or autobiography |
| `hippocampal_reactivate` | retrieve and evaluate guarded lability boundaries |
| `hippocampal_reconsolidate` | integrate cited evidence with immutable versions |
| `hippocampal_cognitive_status` | inspect contexts, events, source monitoring, lability, and self-recollection metadata |

Tool schemas constrain memory kinds, provenance types, confidence/weight ranges,
and required fields. Actor/session context is used as the default source
reference where available.

Hermes must explicitly invoke these tools; simply generating text does not
silently update durable memory.

## Generic host integration

A non-Hermes host can:

1. create `MemoryStore`;
2. wrap it in `MemoryCoordinator`;
3. ingest provenance-bearing observations;
4. call evidence/conflict/maintenance policy;
5. optionally expose foreground activity with `foreground_turn()`; and
6. use `VaultSynchronizer`, `SleepConsolidator`, or the telemetry app
   independently.

The transcript replay worker additionally understands a SQLite source with
`sessions` and `messages` tables. It sends only user/assistant text to the
configured loopback Ollama endpoint; tool payloads and hidden reasoning are
excluded.

Transcript replay and neural sleep are complementary workers:

- transcript replay ingests finalized conversations, checks citations, creates
  memories, and proposes evidence-gated principles or identity memories;
- neural sleep encodes already accepted memories as engrams and performs the
  bounded NREM/REM-inspired circuit pass.

Enabling neural sleep does not replace transcript ingestion. A complete Hermes
deployment keeps the preemptible transcript worker enabled and schedules neural
sleep separately during an exclusive idle GPU window.
