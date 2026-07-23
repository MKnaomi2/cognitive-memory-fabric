# Contributing

## Principles

Changes must preserve provenance, reversibility, bounded writes, local-only
observability, and separation between lifecycle truth and its projections.

Do not:

- add a memory without provenance;
- overwrite conflicts or superseded history;
- hard-delete as the first forgetting action;
- let models submit synaptic weights;
- add observatory lifecycle-write endpoints; or
- make unbounded vault, telemetry, replay, or circuit operations.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[holographic,obsidian,observatory,dev]"
Set-Location viewer
npm ci
```

CUDA work should use a separate `.venv-neural` with the documented PyTorch
wheel.

## Required validation

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\ruff check src tests integrations
.\.venv\Scripts\python -m compileall -q src integrations

Set-Location viewer
npm audit --audit-level=high
npm run lint
npm test
```

Circuit changes additionally require `circuit-check --device cuda` on supported
hardware and a bounded recorded replay. Vault migration changes require a staged
fixture, acceptance audit, rejected-invalid-cutover test, and rollback check.

## Documentation expectations

Update the relevant architecture document when changing lifecycle equations or
thresholds; schemas or synchronization invariants; neural topology/plasticity;
telemetry formats; integrations and scheduled tasks; or security limits.

Pull requests should explain what changed, why, user/operator impact, failure
behavior, and validation. Keep runtime data and personal vault content out of
commits.
