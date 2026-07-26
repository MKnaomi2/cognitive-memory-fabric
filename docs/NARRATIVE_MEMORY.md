# Evidence-grounded narrative memory

The narrative layer lets Hermes connect a set of memories into a useful story
while preserving the boundary between what was recorded and what was inferred.
It is not a free-running fact generator.

## Recall-time composition

`NarrativeEngine.compose()` starts with bounded symbolic recall, then adds:

1. explicitly derived or conflicting memories;
2. memories from the same temporal context or event; and
3. memories with a qualifying overlap in their stored CA1 signatures.

The engine chooses chronological, thematic, or problem-decision-outcome
structure from the cue, unless the caller requests one. Every rendered claim
has one or more `[m<ID>]` citations and one of three visible labels:

- **Remembered** — directly stored evidence or a recorded relationship.
- **Inference** — a proposed connection, including context and neural overlap.
- **Uncertain** — insufficient evidence to make a stronger statement.

Neural overlap is always described as association, never causation. Open
conflicts retain both source memories and state that the sources disagree.

```powershell
cognitive-memory narrative compose `
  --query "How did the Atlas deployment change?" `
  --structure adaptive
```

Hermes exposes the same operation through `hippocampal_narrative`.

## Sleep-time consolidation

After a completed bounded sleep pass, the system examines recent contexts with
at least three active, sufficiently trusted source memories. A clean context
becomes a draft narrative. It becomes active only after:

- a second supporting sleep pass; or
- explicit helpful feedback while all source memories remain active and
  conflict-free.

Archiving a source or opening a conflict marks an active narrative stale.
Drafts remain inspectable in SQLite and are not projected to Obsidian. Active
narratives are written beneath `Narratives/Active` and link back to their
supporting memory notes. If an already-projected narrative becomes stale, its
note is visibly marked stale and must not be used until revalidated.

```powershell
cognitive-memory narrative list --status draft
cognitive-memory narrative list --status active
cognitive-memory narrative-feedback `
  --thread-id <thread-id> `
  --rating helpful
```

Ratings are accepted only as explicit input. The system never treats continued
conversation or silence as positive feedback.

## Storage and auditability

The lifecycle is stored in five normalized tables:

- `narrative_threads` — stable identity, status, support passes, feedback counts;
- `narrative_versions` — summary, algorithm version, confidence, source hash;
- `narrative_claims` — ordered claim text, label, relation, confidence;
- `narrative_sources` — claim-to-memory evidence links; and
- `narrative_feedback` — explicit ratings tied to a thread or recall audit.

Neural rollout audits store query hashes, both candidate orders, the selected
arm, and a deterministic 0–99 bucket. They do not store the query text.

## Effectiveness gate

Run the private 300-case ablation described in
[`EVALUATION.md`](EVALUATION.md). It compares evidence-only composition with
evidence plus neural associations. A neural rollout may advance only when:

- citation precision and source coverage remain acceptable;
- unsupported claims and unlabeled inferences remain zero;
- the private validation split has no material safety regression; and
- at least 20 explicit user ratings support the behavior.

The recommended sequence is shadow, 10%, 25%, 50%, then 100%. A failed gate
keeps the previous setting; it does not weaken evidence or safety thresholds.
