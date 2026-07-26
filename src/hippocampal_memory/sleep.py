"""Systems-consolidation orchestration for exclusive local-GPU sleep windows."""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .activity import foreground_active
from .circuit import CircuitConfig, TrisynapticCircuit
from .cognition import CognitiveMemorySystem
from .coordination import MemoryCoordinator
from .runtime import ExclusiveSleepWindow
from .store import MemoryStore
from .telemetry import FrameRecorder, TelemetryHub
from .narrative import NarrativeEngine


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
        device: str = "cuda",
        recording_stride: int = 4,
    ) -> None:
        self.store = store
        self.model = model
        self.coordinator = MemoryCoordinator(store)
        self.cognition = CognitiveMemorySystem(store, self.coordinator)
        self.state_root = Path(
            state_root or store.db_path.parent / "neural"
        ).expanduser()
        self.config = circuit_config or CircuitConfig()
        self.telemetry = telemetry
        self.device = device
        self.recording_stride = max(1, min(32, int(recording_stride)))

    def _memory_rows(self, limit: int) -> list[Any]:
        with self.store._lock:
            return self.store._conn.execute(
                """
                SELECT f.fact_id, f.content, f.trust_score, f.salience_score,
                       f.memory_kind, f.context_id, f.event_id, f.sequence_index,
                       b.engram_id, b.neuron_ids_json, b.encoding_version,
                       b.content_sha256, b.ca1_signature_json,
                       t.memory_id time_binding_id,
                       (SELECT window_id FROM reconsolidation_windows w
                        WHERE w.fact_id=f.fact_id AND w.status='labile'
                          AND w.closes_at>CURRENT_TIMESTAMP
                        ORDER BY w.opened_at DESC LIMIT 1) reconsolidation_window_id
                FROM facts f LEFT JOIN engram_bindings b
                  ON b.memory_id=CAST(f.fact_id AS TEXT)
                LEFT JOIN time_cell_bindings t
                  ON t.memory_id=CAST(f.fact_id AS TEXT)
                WHERE f.status='active' AND f.memory_kind IN
                    ('episode','fact','principle','identity')
                ORDER BY
                         (COALESCE(b.encoding_version,'')
                          LIKE 'content-v3-pending:%') DESC,
                         (b.memory_id IS NULL) DESC,
                         (COALESCE(b.encoding_version,'')!='content-v3'
                          OR COALESCE(b.content_sha256,'')=''
                          OR COALESCE(b.ca1_signature_json,'[]')='[]') DESC,
                         (reconsolidation_window_id IS NOT NULL) DESC,
                         f.pinned DESC, f.salience_score DESC,
                         f.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    def _load_circuit(self) -> tuple[TrisynapticCircuit, str | None]:
        with self.store._lock:
            row = self.store._conn.execute(
                """
                SELECT checkpoint_id,path,sha256 FROM neural_checkpoints
                WHERE circuit_version=?
                ORDER BY created_at DESC, rowid DESC LIMIT 1
                """,
                (self.config.version,),
            ).fetchone()
        if row is None:
            return TrisynapticCircuit(self.config, device=self.device), None
        path = Path(str(row["path"])).resolve()
        if not path.is_file():
            raise RuntimeError("registered neural checkpoint is missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not hmac.compare_digest(digest, str(row["sha256"])):
            raise RuntimeError("registered neural checkpoint hash mismatch")
        return (
            TrisynapticCircuit.from_checkpoint(path, device=self.device),
            str(row["checkpoint_id"]),
        )

    def run_once(
        self,
        *,
        max_memories: int = 8,
        nrem_cycles: int = 80,
        rem_cycles: int = 40,
    ) -> dict[str, Any]:
        """Run one preemptible NREM→REM cycle and checkpoint it locally."""
        max_memories = max(1, min(32, int(max_memories)))
        rows = self._memory_rows(max_memories)
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
            circuit, parent_checkpoint_id = self._load_circuit()
            replayed = encoded = frames_written = 0
            pending_memory_ids: list[int] = []
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
                    content_hash = hashlib.sha256(
                        str(row["content"]).encode()
                    ).hexdigest()
                    binding_current = (
                        bool(row["neuron_ids_json"])
                        and row["encoding_version"] == "content-v3"
                        and row["content_sha256"] == content_hash
                        and row["ca1_signature_json"] not in {None, "", "[]"}
                    )
                    if binding_current:
                        neurons = json.loads(row["neuron_ids_json"])
                    else:
                        result = circuit.stimulate_content(
                            str(row["content"]),
                            context_key=str(
                                row["context_id"]
                                or row["event_id"]
                                or f"memory:{memory_id}"
                            ),
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
                            encoding_version=f"content-v3-pending:{session_id}",
                            content_sha256=content_hash,
                            ca1_signature=result["ca1_signature"],
                        )
                        pending_memory_ids.append(memory_id)
                        encoded += 1
                        frames_written += self._write_frames(
                            recorder, result["frames"], memory_id
                        )
                    time_cells = circuit.time_cell_assignment(f"memory:{memory_id}")
                    if not row["time_binding_id"]:
                        phase_by_id = {
                            int(cell_id): float(phase)
                            for cell_id, phase in zip(
                                circuit.time_cell_ids.detach().cpu().tolist(),
                                circuit.time_cell_preferred_phase.detach()
                                .cpu()
                                .tolist(),
                            )
                        }
                        self.coordinator.bind_time_cells(
                            memory_id,
                            time_cells,
                            preferred_phases=[
                                phase_by_id[cell_id] for cell_id in time_cells
                            ],
                            circuit_version=self.config.version,
                            context_id=str(row["context_id"] or ""),
                            event_id=str(row["event_id"] or ""),
                            sequence_index=row["sequence_index"],
                        )
                    reconsolidating = bool(row["reconsolidation_window_id"])
                    context_key = str(
                        row["context_id"] or row["event_id"] or f"memory:{memory_id}"
                    )
                    for phase, cycles in (
                        ("nrem", nrem_cycles),
                        ("rem", rem_cycles),
                    ):
                        result = circuit.sleep_replay(
                            neurons,
                            phase=phase,
                            cycles=cycles,
                            context_key=context_key,
                            reconsolidating=reconsolidating,
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
                        self.store._conn.execute(
                            """
                            UPDATE time_cell_bindings SET
                                last_replayed_at=CURRENT_TIMESTAMP
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
                            "context_id": context_key,
                            "time_cell_count": len(time_cells),
                            "reconsolidation_window_id": row[
                                "reconsolidation_window_id"
                            ],
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
                    "parent_checkpoint_id": parent_checkpoint_id,
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
                        json.dumps(
                            {
                                "recording": str(recording),
                                "parent_checkpoint_id": parent_checkpoint_id,
                            }
                        ),
                    ),
                )
                if pending_memory_ids:
                    placeholders = ",".join("?" for _ in pending_memory_ids)
                    self.store._conn.execute(
                        f"""
                        UPDATE engram_bindings SET encoding_version='content-v3'
                        WHERE memory_id IN ({placeholders})
                          AND encoding_version=?
                        """,
                        tuple(str(item) for item in pending_memory_ids)
                        + (f"content-v3-pending:{session_id}",),
                    )
            narratives = NarrativeEngine(self.store).consolidate_drafts(session_id)
            return {
                "status": "completed",
                "session_id": session_id,
                "encoded": encoded,
                "replayed": replayed,
                "frames": frames_written,
                "recording": str(recording),
                "checkpoint": checkpoint,
                "parent_checkpoint_id": parent_checkpoint_id,
                "narratives": narratives,
            }

    def _write_frames(
        self,
        recorder: FrameRecorder,
        frames: list[dict[str, Any]],
        memory_id: int,
        phase: str = "encoding",
    ) -> int:
        selected = frames[:: self.recording_stride]
        if frames and (not selected or selected[-1] is not frames[-1]):
            selected = [*selected, frames[-1]]
        for frame in selected:
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
        return len(selected)
