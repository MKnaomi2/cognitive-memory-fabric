# Operations

## Environments

Keep the core and CUDA installations isolated. The production neural environment
used by the Windows scripts is `.venv-neural`.

```powershell
python -m venv .venv-neural
.\.venv-neural\Scripts\python -m pip install `
  "torch==2.12.1+cu130" --index-url https://download.pytorch.org/whl/cu130
.\.venv-neural\Scripts\python -m pip install `
  -e ".[holographic,obsidian,observatory,dev]"
Set-Location viewer
npm ci
```

## CLI

Global options include `--home`, `--state-db`, `--model`, and `--ollama-url`.

| Command | Purpose |
|---|---|
| `status` | lifecycle/replay status |
| `run --mode auto|micro|deep|backfill [--shadow]` | transcript replay |
| `pause` / `resume` | maintenance control |
| `history --limit N` | replay-run history |
| `digest` | daily maintenance digest |
| `maintain` | deterministic forgetting maintenance |
| `observatory [--port 8765] [--device cpu|cuda]` | loopback telemetry API |
| `vault-plan --vault PATH` | dry-run projection |
| `vault-sync --vault PATH --limit N [--apply]` | bounded projection |
| `circuit-check --device cuda|cpu` | topology/propagation acceptance probe |
| `sleep --state-root PATH --max-memories N` | one NREM→REM consolidation pass |

Always inspect a vault plan before `--apply`.

## Windows tasks

### Neural Observatory

```powershell
.\scripts\Install-NeuralObservatoryTask.ps1 -Enable
```

`Hermes_Neural_Observatory` starts at interactive logon with restart-on-failure
settings and single-instance protection. The runner:

- checks Python, state database, viewer package, and npm;
- refuses if ports 8765 or 3000 are occupied;
- starts the API first and waits up to 30 seconds for readiness;
- starts the loopback UI;
- writes separate stdout/stderr logs; and
- stops the peer process if either service exits.

### Hippocampal sleep

```powershell
.\scripts\Install-HippocampalSleepTask.ps1 -Enable
```

`Hermes_Hippocampal_Sleep` runs daily at 2:10 AM when the machine has been idle
for 30 minutes. It waits up to three hours for idle, wakes the machine when
permitted, ignores duplicate launches, and has a two-hour execution limit.

The legacy replay task should remain disabled after neural cutover.

## Default Windows paths

```text
Repository          C:\Hermes\hippocampal-memory
Hermes home         %LOCALAPPDATA%\hermes
State database      %LOCALAPPDATA%\hermes\state.db
Obsidian vault      C:\Hermes\Knowledge
Neural root         D:\HermesMemory\neural
Recordings          D:\HermesMemory\neural\recordings
Checkpoints         D:\HermesMemory\neural\checkpoints
Observatory logs    %LOCALAPPDATA%\hermes\logs\neural-observatory
```

All runner paths are parameters and can be changed.

## Acceptance checklist

```powershell
# Python behavior
.\.venv-neural\Scripts\python -m pytest -q
.\.venv-neural\Scripts\ruff check src tests integrations
.\.venv-neural\Scripts\python -m compileall -q src integrations

# Actual circuit contract
.\.venv-neural\Scripts\hippocampal-memory circuit-check --device cuda

# Viewer
Set-Location viewer
npm audit --audit-level=high
npm run lint
npm test

# Runtime endpoints
Invoke-RestMethod http://127.0.0.1:8765/health
Invoke-WebRequest http://localhost:3000
```

A complete live validation should additionally:

1. verify geometry reports 36,864 neurons and 770,048 synapses;
2. connect to `/live`;
3. publish one authenticated local frame;
4. confirm the viewer receives it;
5. load an `.hmrec` file and scrub to its last frame;
6. run the vault integrity audit; and
7. perform a bounded sleep pass that writes both recording and hashed checkpoint.

## Recovery

- **Vault:** staging/cutover retains the previous live vault at the chosen archive
  path. Projection journals contain note-level pre-write backups.
- **Memory:** archives can be queried and restored; maintenance does not
  hard-delete facts.
- **Neural:** recordings are append-only session files; checkpoints are
  content-hashed. A missing viewer does not stop recording.
- **Task failure:** inspect the observatory log directory and Windows Task
  Scheduler result. Reinstalling a task is idempotent because registration uses
  `-Force`.

Back up the SQLite store, vault, neural artifacts, and token/config separately.
Source control contains code only and is not a runtime backup.

## Troubleshooting

### Viewer says telemetry is offline

Check `/health`, port ownership, API stderr, and allowed origin. Both services
must use loopback. The UI still shows fallback geometry when telemetry is
unavailable.

### Sleep refuses the GPU

Check foreground leases, `nvidia-smi`, another sleep worker, and Ollama release.
The refusal is intentional; do not bypass it by killing foreground processes.

### Vault sync reports concurrent edit

The note changed after planning. Re-run `vault-plan`, inspect the human change,
and apply a fresh plan. Do not edit the planned before hash.

### Consolidation is rejected

Inspect evidence count, independent source references, mean confidence,
contradiction ratio, open conflicts, and—for identity—the seven-day time span.
Rejection is policy behavior, not a model failure.
