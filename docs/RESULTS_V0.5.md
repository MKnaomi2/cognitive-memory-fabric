# Initial evaluation results

## Status

These are engineering results, not evidence of general product superiority.
The synthetic challenge sets are small, the symbolic baseline is near ceiling,
and the paired 95% intervals include zero.

## Baseline correction

The first neural run used neural overlap on the raw retrieval score while the
symbolic condition included replay-strength reinforcement. That was not a
proper nested ablation. The raw artifact remains preserved, but it must not be
used to compare symbolic and neural replay.

The evaluator now applies the identical symbolic replay score first and then
lets the neural stage rerank that same ordered candidate pool. It also records
candidate availability, before/after ranks, neural help/harm, overlap margins,
cue/signature sizes, region activity, fallback, and compute cost.

## Development

Fifty-six public synthetic worlds covered fourteen independent challenge
families. The unconstrained neural weight of 0.35 harmed accuracy and was
rejected. A small development grid selected:

- lexical cue;
- neural weight 0.05;
- activation only when the leading overlap is at least 0.70; and
- no minimum overlap-margin threshold.

This candidate was frozen in
[`benchmarks/neural-candidate-v0.5.1.json`](../benchmarks/neural-candidate-v0.5.1.json)
before inspecting holdout outcomes.

| Development result | Symbolic | Frozen neural candidate |
|---|---:|---:|
| Current-answer accuracy | 98.21% | 100.00% |
| Mean reciprocal rank | 0.9911 | 1.0000 |
| Helped / harmed ranks | — | 1 / 0 |
| Mean latency | 27.39 ms | 55.94 ms |
| p95 latency | 94.70 ms | 130.18 ms |

The improvement occurred in paraphrase recall. The absolute accuracy delta was
1.79 percentage points.

## Frozen holdout

The same candidate was then run once on a separately seeded 56-world holdout.

| Holdout result | Symbolic | Frozen neural candidate |
|---|---:|---:|
| Current-answer accuracy | 96.43% | 98.21% |
| Mean reciprocal rank | 0.9821 | 0.9911 |
| Helped / harmed ranks | — | 1 / 0 |
| Mean latency | 26.82 ms | 55.78 ms |
| p95 latency | 95.27 ms | 130.14 ms |

The held-out improvement again occurred in paraphrase recall. The result
replicated directionally, but the paired interval was 0.00 to 5.36 percentage
points and the multiplicity-adjusted result was not significant.

## Decision

The neural hypothesis remains open. The safety gate prevented the severe
regression seen with an overweighted neural score, and the frozen candidate
produced a small directional lift twice. It did not meet the preregistered
minimum three-point delayed/associative improvement, and it roughly doubled
mean retrieval latency in the tiny CPU evaluation.

The next evidence milestone is a larger independently generated replication,
followed by pinned local semantic embeddings and production-GPU trials. No
README or release should claim that neural replay outperforms symbolic replay
until those gates pass.
