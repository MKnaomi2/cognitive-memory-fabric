# Changelog

## Unreleased

### Production neural recall

- Added a persistent authenticated loopback CUDA readout service so Hermes can
  use neural reranking without installing Torch into its runtime.
- Added hash-verified checkpoint loading/reloading, bounded symbolic fallback,
  and privacy-preserving live shadow audits.
- Made neural queries deterministic and non-mutating while avoiding
  Observatory telemetry work, reducing isolated RTX steady-state service
  latency to roughly 53 ms.
- Wired the frozen v0.5.1 cue, weight, margin, and activation safety gate into
  the Hermes provider configuration.
- Made neural sleep resume the latest hash-verified checkpoint and prioritize
  unbound or legacy memories, so repeated passes accumulate learning and
  progressively backfill the live corpus.
- Made newly encoded bindings pending until their checkpoint is registered, so
  an interrupted or size-bounded pass cannot expose checkpointless engrams as
  current.
- Prioritized pending recovery bindings ahead of ordinary unbound memories so
  the next bounded pass repairs interrupted work immediately.
- Sampled recorded sleep telemetry at a bounded stride while retaining final
  frames, allowing the maximum 32-memory pass to remain within the recording
  size limit.

## 0.5.1 — 2026-07-26

### Engineering validation baseline

- Added an isolated Windows RTX engineering-validation harness with strict
  live-path overlap refusal and scheduled-task fingerprinting.
- Added Python 3.11 neural CI requiring `48 passed, 0 skipped`, plus real
  Observatory integration and Playwright coverage.
- Made the viewer API origin configurable for testing while enforcing
  loopback-only origins, and bounded telemetry and MessagePack payloads.
- Updated viewer dependencies to a zero-vulnerability audit result.
- Verified the 36,864-neuron, 770,048-synapse CUDA circuit, neural migration,
  vault synchronization, bounded sleep/checkpoint integrity, and development
  and holdout reproduction.
- Normalized evaluation protocol hashing across LF and CRLF checkouts.
- Observed a small directional neural lift that did not demonstrate
  superiority. The preregistered 1,000-trial publication study is not included.

## 0.5.0 — 2026-07-23

### Evaluation and reproducibility

- Added a frozen five-condition ablation protocol, deterministic synthetic
  worlds, fixed agent scenarios, paired bootstrap intervals, cost/failure
  telemetry, artifact hashes, verification, and mechanically gated claims.
- Added content-derived EC cues, persisted CA1 signatures, checkpoint loading,
  and frozen neural reranking of the same candidates used by symbolic recall.
- Added symbolic replay as the direct non-neural ablation and fail-closed
  symbolic fallback for unavailable or slow neural inference.
- Added a first-class Hermes lifecycle provider with bounded, untrusted
  provenance-aware recall and reversible configuration commands.
- Added CI evaluation smoke runs and documented publication/private-replication
  boundaries. Neural retrieval remains opt-in.
- Added independent development and frozen-holdout challenge families,
  before/after rank diagnostics, conservative neural activation gates, and a
  frozen v0.5.1 candidate. The first holdout showed a small directional lift
  without meeting the preregistered superiority gate.
- Corrected the nested ablation so neural reranking starts from the same
  symbolic replay ordering as the non-neural condition.
- Reworked the Neural Observatory to distinguish functional topology,
  illustrative coordinates, aggregate configured pathways, and exact measured
  activity, with bounded live neuron adjacency.

## 0.4.0 — 2026-07-23

### Cognitive Memory Fabric rebrand

- Renamed the product and distribution to Cognitive Memory Fabric so the public
  identity describes the complete agent-memory platform rather than one neural
  subsystem.
- Named the spiking neural subsystem the Hippocampal Replay Engine and clarified
  that it is an engineering abstraction rather than a claim of biological
  equivalence.
- Added the primary `cognitive-memory` CLI while retaining the former command,
  Python import namespace, state directories, environment variable, and Hermes
  tool identifiers as compatibility interfaces.
- Updated repository paths, Hermes metadata, Obsidian display metadata, viewer
  metadata, operations guidance, and public positioning.

## 0.3.0 — 2026-07-23

### Cognitive temporal memory

- Added temporal contexts, event segmentation, order bindings, recency recall,
  source monitoring, and context reinstatement.
- Added autobiographical self-relevance, perspective, vividness, and
  remember/know/inferred metadata without making a consciousness claim.
- Added guarded reconsolidation boundaries, labile windows, immutable
  before/after versions, and evidence-based restabilization.
- Added context-remapped EC/CA1 time cells, elapsed-phase telemetry/decoding,
  temporal bindings, and a reconsolidation plasticity mode.
- Added Hermes tools, CLI inspection, idempotent legacy-memory enrichment, and
  expanded Obsidian projection.
- Pinned Ruff 0.15.22 as the reproducible lint baseline for this release.

## 0.2.1 — 2026-07-23

### Reliability

- Increased replay context/output budgets and bounded abstraction prompts to
  prevent structured local-model output truncation.
- Isolated malformed source sessions with persisted delayed retries so one
  response cannot block the queue.
- Preserved completed ingestion when a consolidation proposal is malformed.
- Raised the default holographic representation to 4,096 dimensions and
  documented index-capacity expectations.
- Clarified that transcript ingestion and neural sleep are complementary
  scheduled stages.

## 0.2.0 — 2026-07-23

### Memory lifecycle

- Added explicit provenance, confidence evidence, temporal validity, conflict
  preservation, supersession lineage, reversible forgetting, and restoration.
- Added deterministic evidence gates for principle and identity consolidation.
- Added append-only attributed memory events, optimistic revisions,
  idempotency, and payload hashes.

### Obsidian and integration

- Added concept-centric Obsidian projection with preserved human notes,
  bounded writes, atomic replacement, synchronization ledger, and rollback.
- Added transactional full-vault migration, audit, duplicate archival,
  canonical maps, and recovered-link quarantine.
- Added five native Hermes tools and a desktop Obsidian observatory adapter.

### Neural consolidation

- Added the 36,864-neuron, 770,048-synapse EC→DG→CA3→CA1 sparse LIF circuit.
- Added local STDP, homeostasis, refractory dynamics, fixed local inhibition,
  deterministic engram binding, NREM replay, and REM integration.
- Added exclusive-GPU arbitration, Ollama release, foreground preemption,
  MessagePack recordings, and hashed PyTorch checkpoints.

### Observability

- Added loopback telemetry, authenticated publication, binary WebSockets,
  geometry/provenance views, and recording retrieval.
- Added the interactive WebGPU/WebGL Neural Observatory.
- Added Windows scheduled-task installers for observatory and sleep.

## 0.1.0

- Extracted the standalone provenance-aware lifecycle and local replay worker
  from the original Hermes integration.
