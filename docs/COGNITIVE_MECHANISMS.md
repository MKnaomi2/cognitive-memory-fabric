# Cognitive and temporal mechanisms

This package implements computational analogues of memory functions. It does
not claim that software has phenomenal consciousness or that the circuit is a
biologically complete hippocampus.

## Constructs

| Construct | Operational definition | Implementation |
|---|---|---|
| Temporal context | The who/where/when/cue envelope of an experience | `temporal_contexts`; every enriched memory carries context, event interval, and uncertainty |
| Temporal-order memory | Recall of event order independently of database insertion order | `episodic_events`, `event_memories`, and directed `temporal_bindings` with deltas/confidence |
| Recency memory | Relative “how recently” signal at a stated reference time | `recall_recent()` exposes age and configurable half-life decay |
| Source memory | Memory for origin plus an assessment of attribution credibility | immutable provenance/evidence plus versioned `source_monitoring_assessments` |
| Autobiographical memory | Temporally situated, self-relevant episodes | episode/context/event metadata, self relevance, perspective, and an ordered timeline |
| Autonoetic consciousness | Operational “remember/know/inferred” self-recollection metadata | recollection mode, field/observer/semantic perspective, vividness, and self relevance; explicitly not phenomenal consciousness |

## Cognitive processes

### Source monitoring

Source monitoring combines a declared-source prior, attribution completeness,
independent corroboration, confirming-versus-contradicting consistency, and
temporal plausibility. Components, score, decision, and time are persisted. The
assessment never rewrites the original attribution.

### Event segmentation and temporal binding

Memories are sorted by represented occurrence time. A new boundary is created
at a configured temporal gap or sufficiently strong topic shift after a minimum
interval. Boundary reason and algorithm version are stored. Adjacent memories
receive explicit `before` edges, elapsed-time deltas, and confidence.

### Context reinstatement

A direct memory cue selects its bound context; a text cue scores context
summaries and cues. Reinstatement returns context contents in event/sequence
order, records the retrieval, and updates `last_reinstated_at`.

### Consolidation and reconsolidation

Existing evidence gates control principle and identity consolidation.
Consolidated identities are inferred autobiographical meta-memory and retain
derivation links.

Retrieval alone does not automatically destabilize memory. `reactivate()` opens
a one-to-twelve-hour labile window only when prediction error is at least 0.20,
retrieval duration exceeds a threshold that rises with confidence and age, and
the memory is active. The baseline is content-hashed and versioned.
`reconsolidate()` accepts cited evidence and bounded contextual updates, writes
a restabilized version, and closes the window. Unsupported content overwrite is
rejected. Neural sleep gives labile traces a distinct plasticity regime but
does not invent evidence or silently close the window.

## Time cells

The `trisynaptic-v2-time-cells` circuit designates deterministic EC and CA1
neurons as time cells. Each has a preferred normalized elapsed phase, a
receptive field that broadens later in the interval, Gaussian time-dependent
drive, deterministic context remapping, and an inspectable
memory/context/sequence binding.

Sequences run during encoding and NREM/REM replay. Alternating NREM packets
replay temporal phase forward and backward. Telemetry exposes active time-cell
IDs, encoded phase, and decoded phase; geometry and checkpoints store the
definitions. These are simulated neuron dynamics in the LIF circuit, not
decorative labels.

Primary findings informing this engineering model:

- [Umbach et al., PNAS 2020](https://pubmed.ncbi.nlm.nih.gov/33109718/)
- [Reddy et al., Journal of Neuroscience 2021](https://pubmed.ncbi.nlm.nih.gov/34183446/)
- [MacDonald et al., Neuron 2011](https://pubmed.ncbi.nlm.nih.gov/21867888/)
- [Fukushima et al., Learning & Memory 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7167366/)
- [Clem & Huganir, Neuron 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3657785/)
