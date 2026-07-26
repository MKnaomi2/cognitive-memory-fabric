"""Real SQLite/circuit/MessagePack fixture shared by integration and browser tests."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import uvicorn

from hippocampal_memory.circuit import TrisynapticCircuit
from hippocampal_memory.cognition import CognitiveMemorySystem
from hippocampal_memory.coordination import MemoryCoordinator
from hippocampal_memory.store import MemoryStore
from hippocampal_memory.telemetry import FrameRecorder, TelemetryHub, create_app

E2E_TOKEN = "cmf-e2e-test-token-not-a-secret-" + ("0" * 40)


@dataclass
class ObservatoryFixture:
    root: Path
    store: MemoryStore
    circuit: TrisynapticCircuit
    hub: TelemetryHub
    recording: Path
    memory_ids: list[int]

    def app(self):
        return create_app(
            self.store,
            geometry=self.circuit.geometry(),
            hub=self.hub,
            recordings_root=self.root / "recordings",
            publisher_token=E2E_TOKEN,
            neuron_connectivity=self.circuit.neuron_connectivity,
        )


def build_fixture(root: Path) -> ObservatoryFixture:
    root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(root / "state.db")
    coordinator = MemoryCoordinator(store)
    cognition = CognitiveMemorySystem(store, coordinator)
    timestamps = [
        "2026-07-26T01:00:00+00:00",
        "2026-07-26T01:01:00+00:00",
        "2026-07-26T01:02:00+00:00",
    ]
    contents = [
        "Synthetic operator inspected the isolated memory circuit.",
        "Synthetic telemetry crossed the authenticated loopback channel.",
        "Synthetic replay preserved temporal order and source attribution.",
    ]
    memory_ids = [
        coordinator.ingest(
            content,
            actor_type="user" if index == 0 else "agent",
            actor_ref=f"synthetic:observatory:{index}",
            source_uri=f"synthetic://observatory/{index}",
            memory_kind="episode" if index != 1 else "fact",
            valid_from=timestamps[index],
            event_start_at=timestamps[index],
            event_end_at=timestamps[index],
            salience_score=0.9 - index * 0.1,
            source_quality=0.95 - index * 0.1,
            autobiographical=index != 1,
            perspective="field",
            recollection_mode="remember",
        )
        for index, content in enumerate(contents)
    ]
    cognition.segment_memories(memory_ids, context_id="context:synthetic-observatory")
    for memory_id in memory_ids:
        cognition.monitor_source(memory_id)
    coordinator.bind_engram(
        memory_ids[0],
        [0, 1, 2, 3],
        circuit_version="trisynaptic-v2",
        encoding_version="memory-id-v2",
        content_sha256="",
    )

    circuit = TrisynapticCircuit(device="cpu")
    probe = circuit.stimulate_content(
        contents[1],
        context_key="context:synthetic-observatory",
        steps=6,
        plastic=False,
    )
    recording = root / "recordings" / f"sleep-e2e-{os.getpid()}.hmrec"
    with FrameRecorder(recording) as recorder:
        for index, frame in enumerate(probe["frames"][-3:]):
            frame = {
                **frame,
                "step": 9000 + index,
                "memory_id": memory_ids[1],
                "phase": "nrem",
            }
            recorder.append(TelemetryHub().encode(frame))
    return ObservatoryFixture(
        root=root,
        store=store,
        circuit=circuit,
        hub=TelemetryHub(),
        recording=recording,
        memory_ids=memory_ids,
    )


def main() -> None:
    configured = os.environ.get("CMF_E2E_ROOT")
    root = (
        Path(configured).resolve()
        if configured
        else Path(tempfile.mkdtemp(prefix="cmf-observatory-e2e-")).resolve()
    )
    fixture = build_fixture(root)
    port = int(os.environ.get("CMF_E2E_API_PORT", "8766"))
    uvicorn.run(
        fixture.app(),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
