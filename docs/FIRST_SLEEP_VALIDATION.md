# First recorded production sleep pass

On July 24, 2026, the Windows RTX installation completed its first preserved
production sleep pass. This is an engineering milestone, not evidence of
biological fidelity or improved cognition.

## Observed result

- The scheduled task ran from 02:10:01 to 02:11:26 local time and returned
  exit code 0.
- Eight memories were processed without exposing their contents.
- The recording contains 1,200 readable MessagePack frames: 240 encoding,
  640 NREM, and 320 REM.
- Every frame contains nonzero neural and time-cell activity.
- Temporal phase traverses the full normalized range from 0.0 to 1.0 across
  137 distinct encoded phases.
- Peak activity was 2,880 neurons, 12,000 active edges, and 451 time cells.
- Aggregate regional spike totals were EC 52,711, DG 104,042, CA3 156,518,
  and CA1 133,400.
- The sleep session emitted eight `engram.replayed` lifecycle events.
- The 259,635,337-byte `.hmrec` recording is structurally readable.
- The 15,869,253-byte checkpoint SHA-256 matches its database registry entry:
  `0ea5c3c0baf522a2b8a0786eff7610cbe1b830e49b1b13775e9156387a923e75`.

## Interpretation

The pass validates the integrated GPU replay, NREM/REM phase sequencing,
time-cell dynamics, bounded telemetry recording, lifecycle event emission,
checkpoint persistence, and registry integrity on the target Windows system.

It does not establish that sleep replay improves recall, resistance to
interference, generalization, or agent task performance. Those claims require
controlled sleep-versus-no-sleep and symbolic-versus-neural comparisons over
repeated trials. The exact source revision was not embedded in this early
session's recording metadata, so the artifact should be treated as operational
history rather than a fully reproducible scientific result.
