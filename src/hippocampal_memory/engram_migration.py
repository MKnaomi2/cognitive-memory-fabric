"""Explicit migration from ID-derived v2 engrams to content-derived v3."""

from __future__ import annotations

import hashlib
from typing import Any

from .circuit import CircuitConfig, TrisynapticCircuit
from .coordination import MemoryCoordinator
from .store import MemoryStore


class EngramMigrator:
    def __init__(
        self,
        store: MemoryStore,
        *,
        device: str = "cpu",
        config: CircuitConfig | None = None,
    ) -> None:
        self.store = store
        self.device = device
        self.config = config or CircuitConfig()

    def plan(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.store._conn.execute(
            """
            SELECT f.fact_id, f.content, b.encoding_version, b.content_sha256
            FROM facts f JOIN engram_bindings b
              ON b.memory_id=CAST(f.fact_id AS TEXT)
            WHERE b.encoding_version!='content-v3'
               OR b.content_sha256=''
            ORDER BY f.fact_id LIMIT ?
            """,
            (max(1, min(10_000, int(limit))),),
        ).fetchall()
        return [
            {
                "memory_id": int(row["fact_id"]),
                "from": row["encoding_version"],
                "to": "content-v3",
                "content_sha256": hashlib.sha256(
                    str(row["content"]).encode()
                ).hexdigest(),
            }
            for row in rows
        ]

    def apply(self, limit: int = 100) -> dict[str, Any]:
        plan = self.plan(limit)
        if not plan:
            return {"status": "unchanged", "migrated": 0}
        circuit = TrisynapticCircuit(self.config, device=self.device)
        coordinator = MemoryCoordinator(self.store)
        migrated = []
        for item in plan:
            fact = self.store.get_fact(item["memory_id"])
            result = circuit.stimulate_content(
                fact["content"],
                context_key=str(
                    fact.get("context_id")
                    or fact.get("event_id")
                    or f"memory:{item['memory_id']}"
                ),
                steps=30,
                plastic=True,
            )
            coordinator.bind_engram(
                item["memory_id"],
                result["engram_neurons"],
                circuit_version=self.config.version,
                strength=float(fact["trust_score"]),
                encoding_version="content-v3",
                content_sha256=item["content_sha256"],
                ca1_signature=result["ca1_signature"],
            )
            migrated.append(item["memory_id"])
        return {
            "status": "completed",
            "migrated": len(migrated),
            "memory_ids": migrated,
            "circuit_version": self.config.version,
        }
