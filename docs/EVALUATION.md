# Evaluation and reproducibility

Version 0.5 evaluates the system as five controlled configurations:

1. Basic Hermes compact memory.
2. Hermes Holographic memory.
3. Cognitive Memory Fabric without replay.
4. Fabric with deterministic symbolic replay.
5. Fabric with frozen spiking readout.

The protocol is frozen in
[`benchmarks/protocol-v0.5.json`](../benchmarks/protocol-v0.5.json). The public
corpus is deterministic and synthetic. Private Hermes-history replication may
emit aggregate metrics only, with a minimum cell size of ten; source text,
prompts, retrieved memories, and session identifiers must not enter public
artifacts.

## Reproduce the CI experiment

```powershell
cognitive-memory evaluate doctor
cognitive-memory evaluate run --profile ci --output .\evaluation-ci
cognitive-memory evaluate verify .\evaluation-ci
```

The CI profile is a contract and determinism smoke test. Its output is never
eligible for a research or product superiority claim.

## Development and frozen holdout

The original v0.5 run remains the baseline. The next phase separates tuning
from evaluation:

```powershell
cognitive-memory evaluate run --profile development --output .\evaluation-dev
cognitive-memory evaluate verify .\evaluation-dev

# Run only after parameters have been frozen.
cognitive-memory evaluate run --profile holdout --output .\evaluation-holdout
cognitive-memory evaluate verify .\evaluation-holdout
```

Each split contains independent deterministic cases across exact recall,
paraphrase, partial and corrupted cues, delayed and associative recall,
temporal order, interference, contradiction, source attribution,
reconsolidation, forgetting, poisoning, and event segmentation. Development
outcomes may guide changes. Holdout outcomes may not.

Every neural trial records whether the expected memory entered the candidate
set; symbolic and final ranks; whether the neural reranker helped, harmed, or
left rank unchanged; positive and strongest-negative overlap; cue and CA1
signature sizes; unique active neurons by region; latency; CPU/GPU time; GPU
memory; storage; and fallback behavior. This separates failures in candidate
generation from failures in neural discrimination.

The circuit exposes three cue paths: lexical, semantic sparse projection, and
hybrid. Semantic vectors must come from an externally pinned local embedding
model; the evaluator never substitutes a fake embedding. The projection itself
is deterministic and bounded.

The preregistered development gate for a neural candidate is:

- at least three percentage points improvement on delayed/associative cases;
- paired 95% interval above zero;
- non-negative mean rank improvement;
- no safety-family regression greater than two percentage points; and
- declared latency, GPU, and storage costs.

Failing a gate means “not demonstrated,” not “the hypothesis is false.”

## Publication run

Use a clean checkout of the frozen protocol tag and immutable model digest:

```powershell
cognitive-memory evaluate agent-run `
  --runner-config .\benchmarks\runner-local.json `
  --model-label local-model@sha256:... `
  --output .\agent-trials.jsonl

cognitive-memory evaluate agent-run `
  --runner-config .\benchmarks\runner-codex.json `
  --model-label codex-terra@version `
  --output .\agent-trials.jsonl `
  --append

cognitive-memory evaluate private-aggregate `
  --input .\private-history-trials.jsonl `
  --output .\private-summary.json

cognitive-memory evaluate run --profile publication `
  --output .\evaluation-publication `
  --agent-results .\agent-trials.jsonl `
  --private-summary .\private-summary.json
cognitive-memory evaluate verify .\evaluation-publication
```

The agent file represents 20 fixed scenarios, five repetitions per condition
and model. The local-model reference and Codex/Terra replication are reported
independently. Missing trials, timeouts, and execution failures count as
failures; infrastructure-invalid attempts remain recorded when rerun.

`agent-run` accepts a JSON command array, never a shell string. `{prompt}` and
`{usage_file}` placeholders are substituted as individual arguments. The
committed example uses Hermes one-shot mode with normal user memory/rules
disabled so only the condition-specific evidence block differs. Pin the exact
provider/model in each real runner config. Interrupted runs can resume with
`--append`; completed model/condition/scenario/repeat keys are not rerun.

`private-aggregate` is the privacy boundary for historical-Hermes replication.
It refuses cells smaller than ten and emits only counts and numeric metrics for
the five conditions. Raw prompts, answers, memory text, session identifiers,
and contact data are never copied into the summary or public release.

The run contains the synthetic dataset, trial JSONL, summary, environment
manifest, hashes, and generated report. Raw publication trials and checkpoints
belong in a versioned GitHub release asset with SHA-256 checksums.

## Metrics and claims

The harness scores current-fact accuracy, source attribution, stale-memory use,
contradiction errors, poisoning success, task completion, false consolidation,
fallbacks, latency, and storage. Paired 95% intervals use 10,000 seeded
bootstrap samples.

Claims are mechanical outcomes of preregistered gates. A failed gate is
reported as “not demonstrated.” Neural replay remains opt-in, and its compute
cost is reported separately from quality.

## Private narrative evaluation

Narrative effectiveness has a separate local-only harness:

```powershell
cognitive-memory evaluate narrative `
  --database C:\path\to\isolated-memory.db `
  --output C:\path\to\private-narrative-development `
  --cases 300 `
  --split development
```

The corpus covers exact, partial, paraphrase, associative, temporal,
event-timeline, project-evolution, thematic, and source-attribution cues. Each
case runs two conditions: evidence-only and evidence plus stored neural
associations. The output reports source coverage, citation precision,
unsupported claims, unlabeled inference, and the paired neural coverage delta.
A deterministic 70/30 case-ID split keeps development and validation separate.
Raw cases and trials are private and must not be committed.

Do not advance beyond shadow mode until the 300-case run has no unsupported
claims or unlabeled inferences, the validation split shows no material safety
regression, and at least 20 explicit user ratings have been collected. Roll out
neural selection in bounded stages (10%, 25%, 50%, then 100%), checking recall
audits and feedback at every stage. Silence is never counted as approval.

## Fairness and boundaries

- Every condition receives the same evidence, query, limits, and context budget.
- Every condition has isolated storage.
- Neural inference is plasticity-disabled and only reranks symbolic candidates.
- Neural failure falls back to symbolic retrieval and remains in the score.
- Recalled content is untrusted data, never an instruction or authorization.
- Results apply only to the recorded data, models, and hardware.
