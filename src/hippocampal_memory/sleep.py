"""Systems-consolidation orchestration for exclusive local-GPU sleep windows."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .activity import foreground_active
from .circuit import CircuitConfig, TrisynapticCircuit
from .coordination import MemoryCoordinator
from .runtime import ExclusiveSleepWindow
from .store import MemoryStore
from .telemetry import FrameRecorder, TelemetryHub


class SleepConsolidator:
    """Encode and replay bounded engrams without model-authored weights."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        model: str = "hermes-local:latest",
        state_root: str | Path | None = None,
        circuit_config: CircuitConfig | None = None,
        telemetry: TelemetryHub | None = None,
    ) -> None:
        self.store = store
        self.model = model
        self.coordinator = MemoryCoordinator(store)
        self.state_root = Path(
            state_root or store.db_path.parent / "neural"
        ).expanduser()
        self.config = circuit_config or CircuitConfig()
        self.telemetry = telemetry

    def run_once(
        self,
        *,
        max_memories: int = 8,
        nrem_cycles: int = 80,
        rem_cycles: int = 40,
    ) -> dict[str, Any]:
        """Run one preemptible NREM→REM cycle and checkpoint it locally."""
        max_memories = max(1, min(32, int(max_memories)))
        with self.store._lock:
            rows = self.store._conn.execute(
                """
                SELECT f.fact_id, f.content, f.trust_score, f.salience_score,
                       f.memory_kind, b.engram_id, b.neuron_ids_json
                FROM facts f LEFT JOIN engram_bindings b
                  ON b.memory_id=CAST(f.fact_id AS TEXT)
                WHERE f.status='active' AND f.memory_kind IN
                    ('episode','fact','principle','identity')
                ORDER BY f.pinned DESC, f.salience_score DESC,
                         f.updated_at DESC
                LIMIT ?
                """,
                (max_memories,),
            ).fetchall()
        if not rows:
            return {"status": "empty", "memories": 0}

        session_id = str(uuid.uuid4())
        recording_root = self.state_root / "recordings"
        checkpoint_root = self.state_root / "checkpoints"
        recording = recording_root / f"sleep-{session_id}.hmrec"
        with ExclusiveSleepWindow(
            model=self.model,
            foreground_active=foreground_active,
        ) as lease:
            circuit = TrisynapticCircuit(self.config, device="cuda")
            replayed = encoded = frames_written = 0
            with FrameRecorder(recording) as recorder:
                for row in rows:
                    if lease.should_preempt():
                        return {
                            "status": "preempted",
                            "session_id": session_id,
                            "encoded": encoded,
                            "replayed": replayed,
                            "recording": str(recording),
                        }
                    memory_id = int(row["fact_id"])
                    if row["neuron_ids_json"]:
                        neurons = json.loads(row["neuron_ids_json"])
                    else:
                        result = circuit.stimulate_engram(
                            f"memory:{memory_id}",
                            steps=30,
                            preempt=lease.should_preempt,
                        )
                        if result["status"] != "completed":
                            return {
                                "status": "preempted",
                                "session_id": session_id,
                                "recording": str(recording),
                            }
                        neurons = result["engram_neurons"]
                        self.coordinator.bind_engram(
                            memory_id,
                            neurons,
                            circuit_version=self.config.version,
                            strength=float(row["trust_score"]),
                        )
                        encoded += 1
                        frames_written += self._write_frames(
                            recorder, result["frames"], memory_id
                        )
                    for phase, cycles in (
                        ("nrem", nrem_cycles),
                        ("rem", rem_cycles),
                    ):
                        result = circuit.sleep_replay(
                            neurons,
                            phase=phase,
                            cycles=cycles,
                            preempt=lease.should_preempt,
                        )
                        frames_written += self._write_frames(
                            recorder, result["frames"], memory_id, phase
                        )
                        if result["status"] != "completed":
                            return {
                                "status": "preempted",
                                "session_id": session_id,
                                "recording": str(recording),
                            }
                    with self.store._lock:
                        self.store._conn.execute(
                            """
                            UPDATE engram_bindings
                            SET replay_count=replay_count+1,
                                last_replayed_at=CURRENT_TIMESTAMP,
                                strength=MIN(1.0, strength + 0.02)
                            WHERE memory_id=?
                            """,
                            (str(memory_id),),
                        )
                    self.coordinator.append_event(
                        f"memory:{memory_id}",
                        "engram.replayed",
                        {
                            "sleep_session_id": session_id,
                            "phases": ["nrem", "rem"],
                            "circuit_version": self.config.version,
                        },
                        actor_type="system",
                        actor_ref="sleep-consolidator",
                        correlation_id=session_id,
                    )
                    replayed += 1

            checkpoint = circuit.checkpoint(
                checkpoint_root / f"checkpoint-{session_id}.pt",
                {
                    "sleep_session_id": session_id,
                    "memory_count": replayed,
                    "created_at": datetime.now().astimezone().isoformat(),
                },
            )
            with self.store._lock:
                self.store._conn.execute(
                    """
                    INSERT INTO neural_checkpoints (
                        checkpoint_id,circuit_version,phase,path,sha256,
                        event_revision,created_at,metadata_json
                    ) VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP,?)
                    """,
                    (
                        session_id,
                        self.config.version,
                        "nrem-rem",
                        checkpoint["path"],
                        checkpoint["sha256"],
                        0,
                        json.dumps({"recording": str(recording)}),
                    ),
                )
            return {
                "status": "completed",
                "session_id": session_id,
                "encoded": encoded,
                "replayed": replayed,
                "frames": frames_written,
                "recording": str(recording),
                "checkpoint": checkpoint,
            }

    def _write_frames(
        self,
        recorder: FrameRecorder,
        frames: list[dict[str, Any]],
        memory_id: int,
        phase: str = "encoding",
    ) -> int:
        for frame in frames:
            frame = {
                **frame,
                "memory_id": memory_id,
                "phase": phase,
            }
            payload = (
                self.telemetry.encode(frame)
                if self.telemetry
                else __import__("msgpack").packb(frame, use_bin_type=True)
            )
            recorder.append(payload)
            if self.telemetry:
                self.telemetry.publish_from_thread(frame)
        return len(frames)
