# Security

## Supported version

The current `0.2.x` line receives security fixes. This is an alpha, single-user,
local-first research system and is not designed for hostile multi-tenant use.

## Trust boundaries

- SQLite, the selected vault, neural artifact directories, and the local Hermes
  home are trusted local storage.
- Memory sources are untrusted evidence and must retain provenance.
- Model output is untrusted proposal data; deterministic policy validates it.
- The observatory browser is read-only for lifecycle state.
- The telemetry publisher is a separate local capability protected by a random
  bearer token.
- Obsidian human notes are user-owned content and are never silently replaced.

## Implemented controls

- loopback-only API binding and origin checks;
- constant-time bearer-token comparison for telemetry publication;
- no lifecycle mutation endpoint in the observatory;
- path containment and filename allowlists for recordings and vault writes;
- bounded telemetry frames, recordings, circuit size, replay batches, and vault
  mutation batches;
- immutable payload-hashed events with optimistic revision checks;
- pre-write hashes, atomic replacement, journals, and rollback for vault sync;
- secret-shaped replay redaction and exclusion of tool payloads/hidden reasoning;
- archive-before-delete lifecycle behavior;
- content hashes for neural checkpoints; and
- ignored runtime databases, credentials, tokens, logs, recordings, checkpoints,
  model files, and configuration.

## Important limitations

- Local users/processes with equivalent filesystem rights can read runtime
  memories and artifacts.
- SQLite content is not encrypted by this package.
- PyTorch checkpoints must be treated as trusted local artifacts; do not load
  checkpoints from untrusted sources.
- The bearer token protects publication capability, not confidentiality from
  other same-user local processes.
- Obsidian plugins execute with desktop-app privileges. Install only reviewed
  plugin code.
- The neural model is not a security decision engine.

## Reporting

Report a suspected vulnerability privately to the repository owner through a
GitHub private vulnerability report or another agreed private channel. Include
the affected version, reproducible steps, impact, required privileges, relevant
paths/endpoints, and a proposed mitigation if known.

Do not include real memories, credentials, tokens, or private vault content.
