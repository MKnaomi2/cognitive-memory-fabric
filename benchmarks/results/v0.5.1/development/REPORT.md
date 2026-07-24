# v0.5 Evaluation Report

Protocol: `eval-v0.5-protocol-1`

## Results

![Condition accuracy](results.svg)

| Condition | Accuracy | Source | Stale use | Poison | p95 latency (ms) |
|---|---:|---:|---:|---:|---:|
| basic | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| holographic | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fabric | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fabric-symbolic | 0.982 | 0.982 | 0.000 | 0.000 | 94.703 |
| fabric-neural | 1.000 | 1.000 | 0.000 | 0.000 | 130.178 |

## Preregistered claims

- CI smoke results are not eligible as publication evidence.

## Reproducibility

- Git commit: `0af02062ce88e6ad2bfdeb0665ad8e17814f1015`
- Dirty worktree: `True`
- Trials SHA-256: `ae35d7dd89c4d19da400bb35beb4363601740b5a8cc1e6e7391e3e220b8f8a27`

These claims apply only to the recorded datasets, models, and hardware.
