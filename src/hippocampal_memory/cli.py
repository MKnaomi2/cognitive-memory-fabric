"""Command-line controls for replay and lifecycle maintenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .circuit import TrisynapticCircuit
from .coordination import MemoryCoordinator
from .replay import DEFAULT_MODEL, DEFAULT_URL, HippocampusEngine, ReplayConfig
from .sleep import SleepConsolidator
from .telemetry import RemoteTelemetry, create_app, observatory_token, serve
from .vault import VaultSynchronizer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hippocampal-memory")
    parser.add_argument("--home", type=Path)
    parser.add_argument("--state-db", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_URL)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    run = commands.add_parser("run")
    run.add_argument(
        "--mode", choices=["auto", "micro", "deep", "backfill"], default="micro"
    )
    run.add_argument("--shadow", action="store_true")
    commands.add_parser("pause")
    commands.add_parser("resume")
    history = commands.add_parser("history")
    history.add_argument("--limit", type=int, default=10)
    commands.add_parser("digest")
    commands.add_parser("maintain")
    observatory = commands.add_parser("observatory")
    observatory.add_argument("--port", type=int, default=8765)
    observatory.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    observatory.add_argument(
        "--recordings-root",
        type=Path,
        default=Path(r"D:\HermesMemory\neural\recordings"),
    )
    vault_plan = commands.add_parser("vault-plan")
    vault_plan.add_argument("--vault", type=Path, required=True)
    vault_sync = commands.add_parser("vault-sync")
    vault_sync.add_argument("--vault", type=Path, required=True)
    vault_sync.add_argument("--limit", type=int, default=25)
    vault_sync.add_argument("--apply", action="store_true")
    circuit_check = commands.add_parser("circuit-check")
    circuit_check.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    sleep = commands.add_parser("sleep")
    sleep.add_argument("--state-root", type=Path)
    sleep.add_argument("--max-memories", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    engine = HippocampusEngine(
        home=args.home,
        state_db=args.state_db,
        config=ReplayConfig(model=args.model, ollama_url=args.ollama_url),
    )
    try:
        if args.command == "status":
            result = engine.store.hippocampus_status()
            result.update(enabled=engine.config.enabled, model=engine.config.model)
        elif args.command == "run":
            result = engine.run(args.mode, shadow=args.shadow)
        elif args.command == "pause":
            engine.store.set_hippocampus_paused(True)
            result = {"paused": True}
        elif args.command == "resume":
            engine.store.set_hippocampus_paused(False)
            result = {"paused": False}
        elif args.command == "history":
            result = {"runs": engine.store.hippocampus_history(args.limit)}
        elif args.command == "digest":
            result = engine.daily_digest()
        elif args.command == "maintain":
            result = engine.store.run_forgetting_maintenance()
        elif args.command == "observatory":
            MemoryCoordinator(engine.store)
            circuit = TrisynapticCircuit(device=args.device)
            token = observatory_token(
                engine.store.db_path.parent / "runtime" / "observatory.token",
                create=True,
            )
            app = create_app(
                engine.store,
                geometry=circuit.geometry(),
                recordings_root=args.recordings_root,
                publisher_token=token,
            )
            serve(app, port=args.port)
            result = {"status": "stopped"}
        elif args.command in {"vault-plan", "vault-sync"}:
            coordinator = MemoryCoordinator(engine.store)
            synchronizer = VaultSynchronizer(
                engine.store, args.vault, coordinator=coordinator
            )
            plan = synchronizer.plan()
            if args.command == "vault-plan" or not args.apply:
                result = {
                    "status": "dry-run",
                    "count": len(plan),
                    "mutations": [
                        {
                            "operation": item.operation,
                            "memory_id": item.memory_id,
                            "relative_path": item.relative_path,
                            "reason": item.reason,
                        }
                        for item in plan
                    ],
                }
            else:
                result = synchronizer.apply(plan[: args.limit], max_mutations=args.limit)
                result["remaining"] = max(0, len(plan) - args.limit)
        elif args.command == "circuit-check":
            circuit = TrisynapticCircuit(device=args.device)
            probe = circuit.stimulate_engram(
                "circuit-acceptance-probe", steps=20, plastic=False
            )
            result = {
                "status": probe["status"],
                "device": str(circuit.device),
                "neurons": circuit.neuron_count,
                "synapses": sum(item.pre.numel() for item in circuit.pathways),
                "engram_neurons": len(probe["engram_neurons"]),
                "last_region_spikes": probe["frames"][-1]["region_spikes"],
            }
        else:
            token = observatory_token(
                engine.store.db_path.parent / "runtime" / "observatory.token"
            )
            result = SleepConsolidator(
                engine.store,
                model=engine.config.model,
                state_root=args.state_root,
                telemetry=RemoteTelemetry(token) if token else None,
            ).run_once(max_memories=args.max_memories)
        print(json.dumps(result, indent=2, default=str))
        return 1 if result.get("status") == "failed" else 0
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
