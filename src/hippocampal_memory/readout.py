"""Causal symbolic and frozen-neural retrieval readouts."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from .circuit import TrisynapticCircuit
from .retrieval import FactRetriever
from .store import MemoryStore


@dataclass(frozen=True)
class ReadoutConfig:
    mode: str = "none"
    candidate_limit: int = 50
    recall_limit: int = 10
    deadline_seconds: float = 2.0
    neural_weight: float = 0.05
    cue_mode: str = "lexical"
    neural_margin_min: float = 0.0
    neural_activation_min: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"none", "symbolic", "neural"}:
            raise ValueError("mode must be none, symbolic, or neural")
        if self.cue_mode not in {"lexical", "semantic", "hybrid"}:
            raise ValueError("cue_mode must be lexical, semantic, or hybrid")
        if not 0 <= self.neural_weight <= 1:
            raise ValueError("neural_weight must be between zero and one")
        if not 0 <= self.neural_margin_min <= 1:
            raise ValueError("neural_margin_min must be between zero and one")
        if not 0 <= self.neural_activation_min <= 1:
            raise ValueError("neural_activation_min must be between zero and one")


class MemoryReadout:
    """Rerank an identical symbolic candidate pool without mutating memory."""

    def __init__(
        self,
        store: MemoryStore,
        config: ReadoutConfig | None = None,
        *,
        circuit: TrisynapticCircuit | None = None,
    ) -> None:
        self.store = store
        self.config = config or ReadoutConfig()
        self.circuit = circuit
        self.retriever = FactRetriever(store)

    def search(
        self, query: str, *, semantic_vector: list[float] | None = None
    ) -> dict[str, Any]:
        started = time.perf_counter()
        candidates = self.retriever.search(query, limit=self.config.candidate_limit)
        symbolic_order = [int(item["fact_id"]) for item in candidates]
        if self.config.mode == "none" or not candidates:
            return self._result(
                candidates, started, "none", symbolic_order=symbolic_order
            )
        bindings = self._bindings(candidates)
        for candidate in candidates:
            binding = bindings.get(str(candidate["fact_id"]), {})
            reinforcement = min(
                1.0,
                float(binding.get("strength", 0.0))
                + 0.02 * int(binding.get("replay_count", 0)),
            )
            candidate["symbolic_replay_score"] = reinforcement
            candidate["score"] = 0.85 * float(candidate["score"]) + 0.15 * reinforcement
        candidates.sort(key=lambda item: item["score"], reverse=True)
        symbolic_replay_order = [int(item["fact_id"]) for item in candidates]
        if self.config.mode == "symbolic":
            return self._result(
                candidates,
                started,
                "symbolic",
                symbolic_order=symbolic_replay_order,
            )
        if self.circuit is None:
            return self._result(
                candidates,
                started,
                "symbolic-fallback",
                True,
                symbolic_order=symbolic_replay_order,
            )
        try:
            query_result = self.circuit.query_content(
                query,
                steps=24,
                context_key="frozen-query",
                cue_mode=self.config.cue_mode,
                semantic_vector=semantic_vector,
            )
            query_ca1 = set(query_result["ca1_signature"])
            if time.perf_counter() - started > self.config.deadline_seconds:
                return self._result(
                    candidates,
                    started,
                    "symbolic-fallback",
                    True,
                    symbolic_order=symbolic_replay_order,
                )
            for candidate in candidates:
                binding = bindings.get(str(candidate["fact_id"]), {})
                candidate_ca1 = set(json.loads(binding.get("ca1_signature_json", "[]")))
                union = query_ca1 | candidate_ca1
                overlap = len(query_ca1 & candidate_ca1) / len(union) if union else 0.0
                candidate["neural_readout_score"] = overlap
                candidate["pre_neural_score"] = float(candidate["score"])
            overlaps = sorted(
                (float(candidate["neural_readout_score"]) for candidate in candidates),
                reverse=True,
            )
            discrimination = (
                overlaps[0] - overlaps[1] if len(overlaps) > 1 else overlaps[0]
            )
            weight = (
                self.config.neural_weight
                if (
                    discrimination >= self.config.neural_margin_min
                    and overlaps[0] >= self.config.neural_activation_min
                )
                else 0.0
            )
            for candidate in candidates:
                candidate["score"] = (1.0 - weight) * float(
                    candidate["pre_neural_score"]
                ) + weight * float(candidate["neural_readout_score"])
            candidates.sort(key=lambda item: item["score"], reverse=True)
            result = self._result(
                candidates,
                started,
                "neural",
                symbolic_order=symbolic_replay_order,
            )
            result["query_diagnostics"] = {
                "cue_mode": query_result.get("cue_mode"),
                "cue_size": query_result.get("cue_size", 0),
                "ca1_signature_size": len(query_ca1),
                "region_active_neurons": query_result.get("region_active_neurons", {}),
                "neural_discrimination": round(discrimination, 6),
                "applied_neural_weight": weight,
                "peak_neural_score": round(overlaps[0], 6),
            }
            return result
        except Exception as exc:
            result = self._result(
                candidates,
                started,
                "symbolic-fallback",
                True,
                symbolic_order=symbolic_replay_order,
            )
            result["error"] = f"{type(exc).__name__}: {exc}"
            return result

    def _bindings(self, candidates: list[dict]) -> dict[str, dict]:
        ids = [str(item["fact_id"]) for item in candidates]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.store._conn.execute(
            f"SELECT * FROM engram_bindings WHERE memory_id IN ({placeholders})",
            ids,
        ).fetchall()
        return {str(row["memory_id"]): dict(row) for row in rows}

    def _result(
        self,
        candidates: list[dict],
        started: float,
        effective_mode: str,
        fallback: bool = False,
        *,
        symbolic_order: list[int] | None = None,
    ) -> dict[str, Any]:
        return {
            "memories": candidates[: self.config.recall_limit],
            "candidate_count": len(candidates),
            "requested_mode": self.config.mode,
            "effective_mode": effective_mode,
            "fallback": fallback,
            "symbolic_order": symbolic_order or [],
            "final_order": [int(item["fact_id"]) for item in candidates],
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
