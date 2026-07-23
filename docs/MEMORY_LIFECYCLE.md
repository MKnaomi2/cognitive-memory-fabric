# Memory lifecycle

## Memory record

Every memory is one of:

- **episode** — a situated occurrence or observation;
- **fact** — a durable claim about the world;
- **principle** — a reusable abstraction derived from multiple observations;
- **identity** — evidence-gated knowledge about the agent's own recurring
  behavior or operating style.

A record carries content, kind, status, provenance, confidence, evidence
counts, validity and expiry windows, relevance, salience, source quality,
pinning, optional subject/predicate keys, and archive/supersession lineage.

## Provenance

Accepted source types are `user`, `agent`, `web`, `reflection`, `sensor`,
`system`, and `imported`. Immutable integration events use `import` for the
corresponding actor.

Provenance has three levels:

1. **Memory origin:** type, reference, source URI, and structured metadata.
2. **Evidence history:** one row per source, confirmation, or contradiction,
   with weight and observation time.
3. **Event attribution:** actor, correlation/causation IDs, event time,
   revision, and a SHA-256 of the canonical payload.

This supports questions such as “Who asserted this?”, “Which later evidence
changed confidence?”, and “Which replay or vault write followed from it?”

## Confidence update

Confidence is clamped to `[0,1]`. Evidence weights are also clamped to `[0,1]`.

For prior confidence `c` and weight `w`:

```text
confirmation:  c′ = c + (1 − c) × 0.15 × w
contradiction: c′ = c − c × 0.25 × w
source:        c′ = c
```

Confirmation has diminishing returns as confidence approaches one.
Contradiction removes a fraction of current confidence, making strong
contradictory evidence consequential without producing a negative score.

Repeated identical content does not create another fact row. It creates
confirming evidence with the new provenance. Helpful retrieval feedback adds
`0.05`; unhelpful feedback subtracts `0.10`.

## Conflict detection

A conflict is a relationship, not an overwrite:

1. Both fact rows remain stored.
2. A `fact_conflicts` row links them and records the reason.
3. Both memories become `conflicted`.
4. Each receives contradictory evidence.
5. Open conflicts block principle/identity consolidation.

Version 0.2 deliberately exposes recording and inspection but no automatic
conflict-resolution method. A host can later implement explicit resolution,
retain a temporal distinction, archive one record, or use the guarded
supersession operation. No implicit winner is selected.

## Supersession

Supersession is used for newer state, not ordinary disagreement. It succeeds
only when:

- old and new memories are different records;
- both have the same non-empty normalized subject key;
- both have the same non-empty normalized predicate key; and
- replacement source quality is at least the old source quality.

The old memory is archived, linked through `superseded_by`, and assigned a
`valid_until` timestamp if it did not already have one. This preserves the fact
that an older value may once have been correct.

## Consolidation

Consolidation creates a principle or identity memory and records
`consolidated_from` derivations to every source. The governed replay worker
retains source episodes and schedules them for archival after a seven-day grace
period. Direct callers of `consolidate()` can instead choose immediate archival
through its `archive_sources` argument.

Eligibility is deterministic:

| Gate | Principle | Identity |
|---|---:|---:|
| Support count | ≥3 | ≥5 |
| Independent source references | ≥2 | ≥3 |
| Mean source confidence | ≥0.70 | ≥0.80 |
| Contradiction ratio | <0.20 | <0.20 |
| Open conflicts | 0 | 0 |
| Evidence time span | none | ≥7 days |

Support count is the greater of qualifying memory rows and distinct source
references. Contradiction ratio is:

```text
contradictions /
max(1, contradictions + confirmations + qualifying memory rows)
```

An LLM may propose a concise abstraction, but cannot waive these gates.

## Identity meta-memory

Identity memory answers “What does repeated evidence say about how this agent
operates?” It intentionally has stricter requirements than a principle:
additional observations, additional independent sources, higher confidence,
and a minimum seven-day span.

Examples of the intended class are recurring planning strengths, persistent
tool-use tendencies, or stable failure modes. A single self-reflection cannot
become identity.

## Forgetting and relevance

Automated forgetting is deterministic and archive-first:

| Rule | Condition |
|---|---|
| `expired` | `expires_at` reached and memory is not pinned |
| `consolidated-grace-complete` | source episode has a derivation, its review date has arrived, and it is not pinned |
| `stale-low-value` | episode older than 180 days, confidence below 0.45, never retrieved, unpinned, no review hold, no open conflict |
| `resolved-low-confidence` | confidence below 0.25, at least three contradictions, unpinned, no open conflict |

Open conflicts are preserved for reasoning. Pinned memories are never selected
by these rules. Restoration moves a memory back to `active`, records
`restored_at`, and applies a 30-day review exemption.

## Decision evidence

Replay and maintenance decisions record:

- action and target;
- source memory IDs;
- reason;
- accepted/rejected status;
- bounded structured payload; and
- decision time.

This makes “why did the agent remember, reject, consolidate, supersede, or
archive this?” answerable without exposing hidden model reasoning.
