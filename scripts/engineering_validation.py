"""Seed and validate isolated migration, vault, sleep, and artifact contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import msgpack

from hippocampal_memory.cognition import CognitiveMemorySystem
from hippocampal_memory.coordination import MemoryCoordinator
from hippocampal_memory.engram_migration import EngramMigrator
from hippocampal_memory.migration import VaultMigrator
from hippocampal_memory.sleep import SleepConsolidator
from hippocampal_memory.store import MemoryStore
from hippocampal_memory.telemetry import read_recording
from hippocampal_memory.vault import VaultSynchronizer


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def seed(store: MemoryStore) -> list[int]:
    coordinator = MemoryCoordinator(store)
    cognition = CognitiveMemorySystem(store, coordinator)
    contents = [
        "Synthetic engineer opened the isolated circuit validation session.",
        "Synthetic loopback telemetry preserved authenticated frame delivery.",
        "Synthetic replay linked the ordered observation to its source.",
    ]
    timestamps = [
        "2026-07-26T02:00:00+00:00",
        "2026-07-26T02:01:00+00:00",
        "2026-07-26T02:02:00+00:00",
    ]
    memory_ids: list[int] = []
    for index, content in enumerate(contents):
        row = store._conn.execute(
            "SELECT fact_id FROM facts WHERE content=?", (content,)
        ).fetchone()
        if row:
            memory_ids.append(int(row["fact_id"]))
            continue
        memory_ids.append(
            coordinator.ingest(
                content,
                actor_type="user" if index == 0 else "agent",
                actor_ref=f"synthetic:engineering:{index}",
                source_uri=f"synthetic://engineering/{index}",
                provenance={"synthetic": True, "validation_index": index},
                memory_kind="episode" if index != 1 else "fact",
                valid_from=timestamps[index],
                event_start_at=timestamps[index],
                event_end_at=timestamps[index],
                salience_score=0.9 - index * 0.1,
                source_quality=0.95 - index * 0.1,
                autobiographical=index != 1,
                self_relevance=0.5,
                perspective="field",
                recollection_mode="remember",
                vividness=0.7,
            )
        )
    cognition.segment_memories(memory_ids, context_id="context:engineering-validation")
    for memory_id in memory_ids:
        cognition.monitor_source(memory_id)
    for index, memory_id in enumerate(memory_ids):
        binding = store._conn.execute(
            "SELECT 1 FROM engram_bindings WHERE memory_id=?",
            (str(memory_id),),
        ).fetchone()
        if binding is None:
            coordinator.bind_engram(
                memory_id,
                [index * 4 + offset for offset in range(4)],
                circuit_version="trisynaptic-v2",
                encoding_version="memory-id-v2",
                content_sha256="",
            )
    return memory_ids


def validate_migration(store: MemoryStore, memory_ids: list[int], device: str) -> dict:
    migrator = EngramMigrator(store, device=device)
    dry_run = migrator.plan(25)
    _require(
        any(item["memory_id"] == memory_ids[0] for item in dry_run),
        "legacy engram was not present in the dry-run plan",
    )
    applied = migrator.apply(25)
    placeholders = ",".join("?" for _ in memory_ids)
    rows = store._conn.execute(
        f"""
        SELECT f.fact_id,f.content,f.context_id,f.event_id,f.sequence_index,
               b.encoding_version,b.content_sha256,b.ca1_signature_json,
               (SELECT COUNT(*) FROM source_monitoring_assessments s
                WHERE s.fact_id=f.fact_id) source_assessments
        FROM facts f JOIN engram_bindings b
          ON b.memory_id=CAST(f.fact_id AS TEXT)
        WHERE f.fact_id IN ({placeholders}) ORDER BY f.fact_id
        """,
        memory_ids,
    ).fetchall()
    _require(len(rows) == 3, "not all synthetic memories have engrams")
    verified = []
    for row in rows:
        expected_hash = hashlib.sha256(str(row["content"]).encode()).hexdigest()
        signature = json.loads(row["ca1_signature_json"])
        _require(row["encoding_version"] == "content-v3", "engram is not content-v3")
        _require(row["content_sha256"] == expected_hash, "content hash mismatch")
        _require(bool(signature), "CA1 signature is empty")
        _require(bool(row["context_id"]) and bool(row["event_id"]), "temporal binding missing")
        _require(row["sequence_index"] is not None, "order binding missing")
        _require(row["source_assessments"] > 0, "source assessment missing")
        verified.append(
            {
                "memory_id": int(row["fact_id"]),
                "content_sha256": expected_hash,
                "ca1_signature_count": len(signature),
                "context_id": row["context_id"],
                "event_id": row["event_id"],
                "sequence_index": int(row["sequence_index"]),
                "source_assessments": int(row["source_assessments"]),
            }
        )
    return {"dry_run": dry_run, "apply": applied, "verified": verified}


def validate_vault(store: MemoryStore, root: Path, memory_ids: list[int]) -> dict:
    vault = root / "vault"
    synchronizer = VaultSynchronizer(
        store, vault, coordinator=MemoryCoordinator(store)
    )
    plan = synchronizer.plan(memory_ids)
    _require(0 < len(plan) <= 25, "vault plan is empty or exceeds the bound")
    result = synchronizer.apply(plan, max_mutations=25)
    staged = root / "evidence" / "vault-audited"
    if staged.exists():
        raise RuntimeError(f"repeat with a fresh validation root: {staged} exists")
    migration = VaultMigrator(vault).stage(staged)
    audit = VaultMigrator.audit(staged)
    _require(audit.valid, f"VaultMigrator.audit failed: {audit}")
    return {
        "planned": len(plan),
        "sync": result,
        "migration": asdict(migration),
        "audit": asdict(audit),
    }


def validate_sleep(store: MemoryStore, root: Path) -> dict:
    result = SleepConsolidator(store, state_root=root).run_once(
        max_memories=3,
        nrem_cycles=8,
        rem_cycles=8,
    )
    _require(result["status"] == "completed", f"sleep did not complete: {result}")
    _require(result["replayed"] >= 1, "sleep replayed no memories")
    recording = Path(result["recording"]).resolve()
    _require(recording.is_relative_to(root), "recording escaped validation root")
    frames = [
        msgpack.unpackb(payload, raw=False) for payload in read_recording(recording)
    ]
    phases = {str(frame.get("phase")) for frame in frames}
    _require({"nrem", "rem"}.issubset(phases), "recording phase did not change")
    active_time_cells = store._conn.execute(
        "SELECT COUNT(*) FROM time_cell_bindings"
    ).fetchone()[0]
    _require(active_time_cells > 0, "sleep produced no time-cell bindings")

    checkpoint = Path(result["checkpoint"]["path"]).resolve()
    _require(checkpoint.is_relative_to(root), "checkpoint escaped validation root")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    registry = store._conn.execute(
        "SELECT sha256 FROM neural_checkpoints WHERE checkpoint_id=?",
        (result["session_id"],),
    ).fetchone()
    _require(registry is not None, "checkpoint registry entry is missing")
    _require(digest == registry["sha256"], "checkpoint registry hash mismatch")
    return {
        **result,
        "recording_frames_read": len(frames),
        "recording_phases": sorted(phases),
        "active_time_cell_bindings": int(active_time_cells),
        "checkpoint_sha256_verified": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("state", "vault", "recordings", "checkpoints", "logs", "evidence"):
        (root / name).mkdir(parents=True, exist_ok=True)
    store = MemoryStore(root / "state" / "state.db")
    try:
        memory_ids = seed(store)
        summary: dict[str, Any] = {
            "status": "completed",
            "root": str(root),
            "database": str(store.db_path.resolve()),
            "memory_ids": memory_ids,
            "migration": validate_migration(store, memory_ids, args.device),
            "vault": validate_vault(store, root, memory_ids),
            "sleep": validate_sleep(store, root),
        }
    finally:
        store.close()
    target = root / "evidence" / "engineering-domain.json"
    target.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
