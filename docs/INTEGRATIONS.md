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

`integrations/hermes/` is deliberately thin. It imports the standalone package
and registers five tools:

| Tool | Operation |
|---|---|
| `hippocampal_remember` | create/reobserve a memory with type, provenance, confidence, and optional subject/predicate/validity data |
| `hippocampal_query` | search active memories or explicitly include archives |
| `hippocampal_evidence` | record weighted source/confirm/contradict evidence and append an event |
| `hippocampal_archive` | reversibly archive a memory with an attributed reason |
| `hippocampal_vault_sync` | dry-run or apply a bounded Obsidian projection |

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
