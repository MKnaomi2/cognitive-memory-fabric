# Neural consolidation

## Scope

The neural subsystem is a sparse spiking engineering model inspired by the
hippocampal trisynaptic circuit and sleep consolidation. It provides local
plasticity, replay, inspectable activity, and durable checkpoints. It is not a
biophysical simulation or a claim that digital engrams reproduce human memory.

Circuit version `trisynaptic-v3-content-readout` adds content-derived entorhinal
cues and a frozen CA1 retrieval readout while retaining context-remapped scalar
temporal receptive fields to deterministic subsets of EC and CA1. Active IDs,
preferred phases, widths, encoded/decoded elapsed phase, and
memory/context/sequence bindings are inspectable. Alternating NREM packets
provide forward and reverse temporal replay. A labile reconsolidation trace
receives a distinct bounded neuromodulation regime.

## Topology

| Pathway | Source neurons | Fan-out | Synapses | Plastic |
|---|---:|---:|---:|---|
| EC→DG | 8,192 | 8 | 65,536 | yes |
| DG→CA3 | 16,384 | 6 | 98,304 | yes |
| EC→CA3 | 8,192 | 4 | 32,768 | yes |
| CA3→CA3 | 8,192 | 4 | 32,768 | yes |
| CA3→CA1 | 8,192 | 8 | 65,536 | yes |
| EC→CA1 | 8,192 | 4 | 32,768 | yes |
| EC local inhibition | 8,192 | 12 | 98,304 | no |
| DG local inhibition | 16,384 | 12 | 196,608 | no |
| CA3 local inhibition | 8,192 | 12 | 98,304 | no |
| CA1 local inhibition | 4,096 | 12 | 49,152 | no |
| **Total** | **36,864** | | **770,048** | |

Connections are reproducibly generated from seed `41`. Self-connections are
removed for recurrent fields. Initial excitatory and inhibitory weights are
sampled around `0.58` and `0.72` respectively.

## Neuron dynamics

The circuit uses leaky integrate-and-fire neurons:

- simulation step: 1 ms;
- membrane time constant: 20 ms;
- threshold: 1.0;
- reset voltage: 0.0;
- refractory period: 2 ms;
- trace time constant: 20 ms.

At each step, sparse pathway contributions are accumulated at postsynaptic
indices. Inhibitory contributions are negated. Available membrane voltage leaks
by `dt/tau`, receives current, spikes at its adaptive threshold, then resets.

## Plasticity

Plastic pathways use asymmetric local spike-timing-dependent plasticity:

- potentiation coefficient `a+ = 0.006`;
- depression coefficient `a− = 0.007`;
- pre/post traces decay with a 20 ms time constant;
- neuromodulation is bounded to `[0,2]`; and
- weights are clamped to `[0,1]`.

No language model or telemetry client may supply weights. Models can help
select memories for replay; the circuit alone computes weight changes.

Homeostasis tracks a firing-rate exponential moving average with target `0.02`.
Thresholds adapt at rate `0.001` and remain within `[0.6,1.8]`. Fixed local
inhibition and threshold adaptation prevent uncontrolled activity.

## Engram encoding

An engram cue is derived deterministically from normalized content tokens.
Each token hashes into a sparse EC assembly; overlapping content therefore
shares cue cells without retaining query text in the checkpoint metadata:

1. SHA-256 supplies a stable per-memory random seed.
2. Approximately 1.2% of EC neurons are selected, with a minimum of eight.
3. Encoding stimulates the cue for the first eight steps and every eleventh
   later step.
4. All neurons that fire during encoding form the stored engram binding.
5. Binding records circuit version, strength, replay count, and timestamps.

This creates a reproducible index into a plastic distributed circuit without
encoding memory text into telemetry.

## Sleep phases

One default consolidation pass selects at most eight active memories, ordered
by pinning, salience, and recency. The hard bound is 32.

### NREM

Default NREM replay runs 80 cycles. A sharp-wave-like cue packet occurs for four
steps out of every 25. Its amplitude is nested in:

```text
spindle = 0.7 + 0.3 × sin(2π × step / 100)
```

The cue receives `1.1 × spindle`; neuromodulation is `1.0`.

### REM

Default REM replay runs 40 cycles. Every seventeenth interval stimulates half
the engram for three steps at amplitude `0.82`. Neuromodulation is reduced to
`0.55`, encouraging weaker distributed association.

Each successful memory replay increments its count and increases binding
strength by `0.02`, capped at `1.0`. A correlated immutable event records the
sleep session, phases, and circuit version.

## Exclusive GPU window

`ExclusiveSleepWindow`:

1. refuses concurrent sleep windows;
2. refuses entry during an active foreground lease;
3. asks loopback Ollama to unload the configured model;
4. waits up to eight seconds for GPU use to fall below 1,024 MiB;
5. polls foreground activity every 0.5 seconds; and
6. exposes a preemption signal checked between encoding/replay steps.

Ollama remains on-demand and reloads its model with the next inference request.
The worker does not kill foreground GPU processes.

## Durability

Each successful pass writes:

- an `HMREC1` bounded binary recording containing MessagePack frames;
- a PyTorch checkpoint written through temporary-file replacement;
- a SHA-256 of the final checkpoint;
- a `neural_checkpoints` registry row; and
- per-memory replay events and binding updates.

The checkpoint includes configuration, simulation step, voltage, adaptive
thresholds, rate estimates, every pathway's indices/weights/flags, and bounded
session metadata.

Each pass resumes the latest compatible checkpoint only after its registered
SHA-256 is verified. Selection prioritizes bindings left explicitly pending by
an interrupted pass, then active memories without an engram, then legacy or
incomplete bindings, before replaying already current engrams.
The new checkpoint records its parent checkpoint ID, making cumulative sleep
lineage explicit and allowing repeated bounded passes to backfill the corpus.
New or repaired engrams remain marked with a session-specific pending encoding
version until that pass registers its checkpoint. If a pass is interrupted or
exceeds a safety bound, a later pass therefore re-encodes those memories from
the last durable checkpoint instead of treating checkpointless state as
current.

## Safety bounds

- circuit configurations above 250,000 neurons are rejected;
- telemetry active edges are capped at 12,000 per frame;
- recordings default to 512 MiB maximum;
- sleep recordings sample simulation frames at a bounded stride and always
  retain the last frame in each encoding or replay phase;
- individual recorded/ingested frames are capped at 16 MiB;
- recording creation is exclusive (`xb`) to prevent silent overwrite; and
- checkpoint files are content-hashed.

## Frozen retrieval readout

Encoding persists a versioned CA1 spike signature and content hash with each
engram. Query inference uses a content-derived EC cue with plasticity disabled,
then scores overlap between the query CA1 response and each candidate signature.
It can only rerank the same bounded pool returned by symbolic retrieval.

Missing, incompatible, corrupt, or over-deadline neural state fails closed to
symbolic ranking. Fallbacks are observable and count in evaluation results.
Version-2 recordings remain viewable, but version-2 and version-3 checkpoints
must never be mixed in an evaluation.
