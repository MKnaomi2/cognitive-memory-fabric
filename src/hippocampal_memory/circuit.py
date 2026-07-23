"""Sparse, locally plastic EC→DG→CA3→CA1 spiking memory circuit."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import torch
except ImportError:  # Core lifecycle remains usable without neural extras.
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CircuitConfig:
    """A 36,864-neuron circuit sized for the local 16 GB RTX 5060 Ti."""

    populations: dict[str, int] = field(
        default_factory=lambda: {
            "EC": 8192,
            "DG": 16384,
            "CA3": 8192,
            "CA1": 4096,
        }
    )
    fanout: dict[str, int] = field(
        default_factory=lambda: {
            "EC_DG": 8,
            "DG_CA3": 6,
            "EC_CA3": 4,
            "CA3_CA3": 4,
            "CA3_CA1": 8,
            "EC_CA1": 4,
            "LOCAL_INHIBITION": 12,
        }
    )
    dt_ms: float = 1.0
    membrane_tau_ms: float = 20.0
    trace_tau_ms: float = 20.0
    refractory_ms: int = 2
    threshold: float = 1.0
    reset_voltage: float = 0.0
    target_rate: float = 0.02
    homeostasis_rate: float = 0.001
    a_plus: float = 0.006
    a_minus: float = 0.007
    weight_min: float = 0.0
    weight_max: float = 1.0
    time_cell_fraction: float = 0.0625
    time_cell_min_width: float = 0.025
    time_cell_max_width: float = 0.080
    time_cell_current: float = 1.15
    seed: int = 41
    version: str = "trisynaptic-v2-time-cells"

    def validate(self) -> None:
        if set(self.populations) != {"EC", "DG", "CA3", "CA1"}:
            raise ValueError("populations must be EC, DG, CA3, and CA1")
        if any(value <= 0 for value in self.populations.values()):
            raise ValueError("population sizes must be positive")
        if sum(self.populations.values()) > 250_000:
            raise ValueError("circuit is above the supported safety bound")
        if self.dt_ms <= 0 or self.membrane_tau_ms <= self.dt_ms:
            raise ValueError("invalid integration time constants")
        if not 0 <= self.weight_min < self.weight_max:
            raise ValueError("invalid weight bounds")
        if not 0 < self.time_cell_fraction <= 0.25:
            raise ValueError("time_cell_fraction must be in (0, 0.25]")
        if not 0 < self.time_cell_min_width <= self.time_cell_max_width < 0.5:
            raise ValueError("invalid time-cell receptive-field widths")


@dataclass
class SparsePathway:
    name: str
    pre: Any
    post: Any
    weight: Any
    inhibitory: bool = False
    plastic: bool = True


class TrisynapticCircuit:
    """Leaky integrate-and-fire neurons with sparse STDP and homeostasis.

    All synaptic changes are computed from local spike traces. External models
    can choose which memories to replay, but cannot submit or overwrite weights.
    """

    def __init__(
        self,
        config: CircuitConfig | None = None,
        *,
        device: str | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required; install the 'neural' extra")
        self.config = config or CircuitConfig()
        self.config.validate()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for name, count in self.config.populations.items():
            self.offsets[name] = (cursor, cursor + count)
            cursor += count
        self.neuron_count = cursor
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.config.seed)
        self.voltage = torch.zeros(self.neuron_count, device=self.device)
        self.spikes = torch.zeros(
            self.neuron_count, dtype=torch.bool, device=self.device
        )
        self.pre_trace = torch.zeros(self.neuron_count, device=self.device)
        self.post_trace = torch.zeros(self.neuron_count, device=self.device)
        self.thresholds = torch.full(
            (self.neuron_count,), self.config.threshold, device=self.device
        )
        self.rate_ema = torch.zeros(self.neuron_count, device=self.device)
        self.refractory = torch.zeros(
            self.neuron_count, dtype=torch.int16, device=self.device
        )
        self.step_index = 0
        self.pathways = self._build_pathways()
        self.positions = self._build_positions()
        (
            self.time_cell_ids,
            self.time_cell_preferred_phase,
            self.time_cell_width,
        ) = self._build_time_cells()
        self.time_cell_mask = torch.zeros(
            self.neuron_count, dtype=torch.bool, device=self.device
        )
        self.time_cell_mask[self.time_cell_ids] = True
        self.temporal_phase = 0.0

    def _range(self, region: str) -> tuple[int, int]:
        return self.offsets[region]

    def _connect(
        self,
        name: str,
        source: str,
        destination: str,
        fanout: int,
        *,
        inhibitory: bool = False,
        plastic: bool = True,
    ) -> SparsePathway:
        source_start, source_end = self._range(source)
        target_start, target_end = self._range(destination)
        sources = torch.arange(source_start, source_end).repeat_interleave(fanout)
        targets = torch.randint(
            target_start,
            target_end,
            (sources.numel(),),
            generator=self.generator,
        )
        if source == destination:
            targets = torch.where(
                targets == sources,
                target_start + ((targets - target_start + 1) % (target_end - target_start)),
                targets,
            )
        # Sparse fan-out needs a strong unitary EPSP so two temporally aligned
        # inputs can cross threshold; local inhibition keeps the result sparse.
        initial = 0.72 if inhibitory else 0.58
        weights = torch.empty(sources.numel()).uniform_(
            initial * 0.75, initial * 1.25, generator=self.generator
        )
        return SparsePathway(
            name,
            sources.to(self.device),
            targets.to(self.device),
            weights.to(self.device),
            inhibitory,
            plastic,
        )

    def _build_pathways(self) -> list[SparsePathway]:
        fanout = self.config.fanout
        pathways = [
            self._connect("EC_DG", "EC", "DG", fanout["EC_DG"]),
            self._connect("DG_CA3", "DG", "CA3", fanout["DG_CA3"]),
            self._connect("EC_CA3", "EC", "CA3", fanout["EC_CA3"]),
            self._connect("CA3_CA3", "CA3", "CA3", fanout["CA3_CA3"]),
            self._connect("CA3_CA1", "CA3", "CA1", fanout["CA3_CA1"]),
            self._connect("EC_CA1", "EC", "CA1", fanout["EC_CA1"]),
        ]
        for region in self.config.populations:
            pathways.append(
                self._connect(
                    f"{region}_INHIBITION",
                    region,
                    region,
                    fanout["LOCAL_INHIBITION"],
                    inhibitory=True,
                    plastic=False,
                )
            )
        return pathways

    def _build_positions(self) -> Any:
        centers = {
            "EC": (-3.4, 0.0, 0.0),
            "DG": (-1.2, 0.3, 0.8),
            "CA3": (1.0, 0.5, 0.2),
            "CA1": (3.2, 0.0, -0.4),
        }
        positions = torch.empty((self.neuron_count, 3))
        for region, (start, end) in self.offsets.items():
            count = end - start
            theta = torch.rand(count, generator=self.generator) * (2 * math.pi)
            radius = 0.35 + torch.rand(count, generator=self.generator) * 0.9
            noise = torch.randn(count, generator=self.generator) * 0.22
            center = centers[region]
            positions[start:end, 0] = center[0] + torch.cos(theta) * radius
            positions[start:end, 1] = center[1] + torch.sin(theta) * radius
            positions[start:end, 2] = center[2] + noise
        return positions

    def _build_time_cells(self) -> tuple[Any, Any, Any]:
        """Designate EC/CA1 cells with scalar temporal receptive fields."""
        selected = []
        preferred_by_region = []
        for region in ("EC", "CA1"):
            start, end = self._range(region)
            count = max(16, int((end - start) * self.config.time_cell_fraction))
            selected.append(
                torch.linspace(start, end - 1, count, dtype=torch.long)
            )
            preferred_by_region.append(torch.linspace(0.0, 1.0, count))
        ids = torch.cat(selected).to(self.device)
        preferred = torch.cat(preferred_by_region).to(self.device)
        widths = self.config.time_cell_min_width + preferred * (
            self.config.time_cell_max_width - self.config.time_cell_min_width
        )
        return ids, preferred, widths

    def temporal_current(
        self, elapsed_step: int, total_steps: int, *, context_key: str = ""
    ) -> Any:
        """Generate a context-remapped sequence of time-cell activity."""
        phase = max(0.0, min(1.0, elapsed_step / max(1, total_steps - 1)))
        digest = hashlib.sha256(context_key.encode()).digest()
        offset = int.from_bytes(digest[:4], "little") / (2**32)
        remapped = (self.time_cell_preferred_phase + offset) % 1.0
        distance = torch.abs(remapped - phase)
        distance = torch.minimum(distance, 1.0 - distance)
        drive = torch.exp(-0.5 * (distance / self.time_cell_width) ** 2)
        current = torch.zeros(self.neuron_count, device=self.device)
        current[self.time_cell_ids] = drive * self.config.time_cell_current
        self.temporal_phase = phase
        return current

    def time_cell_assignment(self, key: str, count: int = 24) -> list[int]:
        """Return the deterministic temporal assembly bound to one memory."""
        digest = hashlib.sha256(key.encode()).digest()
        center = int.from_bytes(digest[:4], "little") % self.time_cell_ids.numel()
        half = max(1, min(int(count), self.time_cell_ids.numel())) // 2
        indices = [
            (center + delta) % self.time_cell_ids.numel()
            for delta in range(-half, half + 1)
        ]
        return sorted(
            {int(self.time_cell_ids[index].detach().cpu().item()) for index in indices}
        )

    def decode_elapsed_phase(self) -> float | None:
        """Decode elapsed phase from the currently firing time-cell population."""
        active = self.spikes[self.time_cell_ids].float()
        if not bool(active.any()):
            return None
        value = (active * self.time_cell_preferred_phase).sum() / active.sum()
        return round(float(value.detach().cpu().item()), 6)

    def step(
        self,
        external_current: Any | None = None,
        *,
        plastic: bool = True,
        neuromodulation: float = 1.0,
    ) -> dict[str, Any]:
        """Advance one millisecond and return a bounded telemetry frame."""
        current = torch.zeros(self.neuron_count, device=self.device)
        for pathway in self.pathways:
            contribution = pathway.weight * self.spikes[pathway.pre].float()
            if pathway.inhibitory:
                contribution = -contribution
            current.index_add_(0, pathway.post, contribution)
        if external_current is not None:
            external = torch.as_tensor(
                external_current, device=self.device, dtype=current.dtype
            )
            if external.shape != current.shape:
                raise ValueError("external_current must have one value per neuron")
            current.add_(external)

        leak = self.config.dt_ms / self.config.membrane_tau_ms
        available = self.refractory <= 0
        self.voltage = torch.where(
            available,
            self.voltage * (1.0 - leak) + current,
            torch.full_like(self.voltage, self.config.reset_voltage),
        )
        next_spikes = available & (self.voltage >= self.thresholds)
        self.voltage[next_spikes] = self.config.reset_voltage
        self.refractory.sub_(1).clamp_(min=0)
        self.refractory[next_spikes] = self.config.refractory_ms

        trace_decay = math.exp(-self.config.dt_ms / self.config.trace_tau_ms)
        self.pre_trace.mul_(trace_decay).add_(self.spikes.float())
        self.post_trace.mul_(trace_decay).add_(next_spikes.float())
        if plastic:
            modulation = max(0.0, min(2.0, float(neuromodulation)))
            for pathway in self.pathways:
                if not pathway.plastic:
                    continue
                potentiation = (
                    self.config.a_plus
                    * self.pre_trace[pathway.pre]
                    * next_spikes[pathway.post].float()
                )
                depression = (
                    self.config.a_minus
                    * self.spikes[pathway.pre].float()
                    * self.post_trace[pathway.post]
                )
                pathway.weight.add_(modulation * (potentiation - depression))
                pathway.weight.clamp_(
                    self.config.weight_min, self.config.weight_max
                )

        self.rate_ema.mul_(0.995).add_(next_spikes.float(), alpha=0.005)
        self.thresholds.add_(
            self.config.homeostasis_rate
            * (self.rate_ema - self.config.target_rate)
        ).clamp_(0.6, 1.8)
        self.spikes = next_spikes
        self.step_index += 1
        active = torch.nonzero(next_spikes, as_tuple=False).flatten()
        return self.telemetry(active)

    def telemetry(self, active: Any | None = None, edge_limit: int = 12_000) -> dict:
        active = (
            torch.nonzero(self.spikes, as_tuple=False).flatten()
            if active is None
            else active
        )
        active_set = torch.zeros(
            self.neuron_count, dtype=torch.bool, device=self.device
        )
        active_set[active] = True
        edges = []
        for pathway in self.pathways:
            mask = active_set[pathway.pre] | active_set[pathway.post]
            selected = torch.nonzero(mask, as_tuple=False).flatten()[:edge_limit]
            if selected.numel():
                triples = torch.stack(
                    (
                        pathway.pre[selected],
                        pathway.post[selected],
                        pathway.weight[selected],
                    ),
                    dim=1,
                ).detach().cpu().tolist()
                edges.extend(
                    (int(pre), int(post), round(float(weight), 5), pathway.name)
                    for pre, post, weight in triples
                )
            if len(edges) >= edge_limit:
                break
        counts = {
            region: int(
                self.spikes[start:end].sum().detach().cpu().item()
            )
            for region, (start, end) in self.offsets.items()
        }
        active_time_cells = active[self.time_cell_mask[active]]
        return {
            "schema": 1,
            "step": self.step_index,
            "active_neurons": active.detach().cpu().tolist(),
            "active_edges": edges[:edge_limit],
            "region_spikes": counts,
            "time_cells_active": active_time_cells.detach().cpu().tolist(),
            "temporal_phase": round(float(self.temporal_phase), 6),
            "decoded_temporal_phase": self.decode_elapsed_phase(),
        }

    def stimulate_engram(
        self,
        key: str,
        *,
        steps: int = 40,
        ec_fraction: float = 0.012,
        plastic: bool = True,
        preempt: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Encode or replay a deterministic sparse EC cue."""
        start, end = self._range("EC")
        count = max(8, int((end - start) * ec_fraction))
        digest = hashlib.sha256(key.encode()).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF)
        selected = (
            torch.randperm(end - start, generator=generator)[:count] + start
        ).to(self.device)
        peak: set[int] = set()
        frames = []
        for index in range(steps):
            if preempt and preempt():
                return {
                    "status": "preempted",
                    "steps": index,
                    "engram_neurons": sorted(peak),
                    "frames": frames,
                }
            current = self.temporal_current(index, steps, context_key=key)
            if index < 8 or index % 11 == 0:
                current[selected] = 1.35
            frame = self.step(current, plastic=plastic)
            peak.update(int(value) for value in frame["active_neurons"])
            frames.append(frame)
        return {
            "status": "completed",
            "steps": steps,
            "engram_neurons": sorted(peak),
            "time_cell_neurons": self.time_cell_assignment(key),
            "frames": frames,
        }

    def sleep_replay(
        self,
        engram_neurons: list[int],
        *,
        phase: str = "nrem",
        cycles: int = 120,
        context_key: str = "",
        reconsolidating: bool = False,
        preempt: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Replay an engram with NREM ripple/spindle or REM integration timing."""
        if phase not in {"nrem", "rem"}:
            raise ValueError("phase must be nrem or rem")
        cue = torch.as_tensor(
            [value for value in engram_neurons if 0 <= value < self.neuron_count],
            dtype=torch.long,
            device=self.device,
        )
        frames = []
        for index in range(cycles):
            if preempt and preempt():
                return {"status": "preempted", "phase": phase, "frames": frames}
            temporal_index = (
                cycles - index - 1
                if phase == "nrem" and (index // 25) % 2
                else index
            )
            current = self.temporal_current(
                temporal_index, cycles, context_key=context_key
            )
            if cue.numel():
                if phase == "nrem":
                    # Sharp-wave ripple packets nested in a slower spindle.
                    ripple = index % 25 < 4
                    spindle = 0.7 + 0.3 * math.sin(2 * math.pi * index / 100)
                    if ripple:
                        current[cue] = 1.1 * spindle
                else:
                    # Weaker, distributed REM cue supports association without
                    # letting a single trace dominate.
                    if index % 17 < 3:
                        current[cue[::2]] = 0.82
            frames.append(
                self.step(
                    current,
                    plastic=True,
                    neuromodulation=(
                        (1.2 if phase == "nrem" else 0.70)
                        if reconsolidating
                        else (1.0 if phase == "nrem" else 0.55)
                    ),
                )
            )
        return {"status": "completed", "phase": phase, "frames": frames}

    def checkpoint(self, path: str | Path, metadata: dict | None = None) -> dict:
        """Write a local checkpoint and return its content hash."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format_version": 1,
            "config": asdict(self.config),
            "step_index": self.step_index,
            "voltage": self.voltage.detach().cpu(),
            "thresholds": self.thresholds.detach().cpu(),
            "rate_ema": self.rate_ema.detach().cpu(),
            "time_cell_ids": self.time_cell_ids.detach().cpu(),
            "time_cell_preferred_phase": self.time_cell_preferred_phase.detach().cpu(),
            "time_cell_width": self.time_cell_width.detach().cpu(),
            "pathways": {
                item.name: {
                    "pre": item.pre.detach().cpu(),
                    "post": item.post.detach().cpu(),
                    "weight": item.weight.detach().cpu(),
                    "inhibitory": item.inhibitory,
                    "plastic": item.plastic,
                }
                for item in self.pathways
            },
            "metadata_json": json.dumps(metadata or {}, sort_keys=True),
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        return {
            "path": str(destination),
            "sha256": digest,
            "step": self.step_index,
            "circuit_version": self.config.version,
        }

    def geometry(self) -> dict[str, Any]:
        """Static actual-neuron geometry, separated from live telemetry."""
        return {
            "schema": 1,
            "circuit_version": self.config.version,
            "neuron_count": self.neuron_count,
            "positions": self.positions.tolist(),
            "regions": {
                name: {"start": start, "end": end}
                for name, (start, end) in self.offsets.items()
            },
            "pathways": [
                {
                    "name": item.name,
                    "synapse_count": int(item.pre.numel()),
                    "inhibitory": item.inhibitory,
                    "plastic": item.plastic,
                }
                for item in self.pathways
            ],
            "time_cells": {
                "ids": self.time_cell_ids.detach().cpu().tolist(),
                "preferred_phase": self.time_cell_preferred_phase.detach()
                .cpu()
                .tolist(),
                "width": self.time_cell_width.detach().cpu().tolist(),
                "mechanism": "context-remapped scalar temporal receptive fields",
            },
        }
