# v0.5.1 neural candidate results

These are complete public synthetic artifacts for the development-selected
candidate frozen in
[`../../neural-candidate-v0.5.1.json`](../../neural-candidate-v0.5.1.json).

- [`development/`](development/) contains the 56-world tuning run.
- [`holdout/`](holdout/) contains the separately seeded 56-world frozen run.

Each directory contains the generated dataset, per-trial JSONL, summary,
environment/configuration manifest, SVG chart, and report. Verify a copied
directory with:

```powershell
cognitive-memory evaluate verify .\benchmarks\results\v0.5.1\holdout
```

These runs are not publication-eligible and do not satisfy the preregistered
neural superiority gate. See
[`docs/RESULTS_V0.5.md`](../../../docs/RESULTS_V0.5.md) for interpretation.
