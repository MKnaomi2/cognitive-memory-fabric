"""Command-line controls for replay and lifecycle maintenance."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from .circuit import TrisynapticCircuit
from .cognition import CognitiveMemorySystem
from .coordination import MemoryCoordinator
from .replay import DEFAULT_MODEL, DEFAULT_URL, HippocampusEngine, ReplayConfig
from .sleep import SleepConsolidator
from .telemetry import RemoteTelemetry, create_app, observatory_token, serve
from .vault import VaultSynchronizer
from .evaluation import (
    aggregate_private_results,
    CONDITIONS,
    doctor as evaluation_doctor,
    report_evaluation,
    run_agent_trials,
    run_evaluation,
    verify_evaluation,
)
from .hermes_setup import doctor as hermes_doctor
from .hermes_setup import install as hermes_install
from .hermes_setup import uninstall as hermes_uninstall
from .engram_migration import EngramMigrator
from .neural_service import NeuralReadoutRuntime, create_neural_readout_app
from .readout import ReadoutConfig


def _parser(*, prog: str = "cognitive-memory") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
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
    neural_serve = commands.add_parser("neural-serve")
    neural_serve.add_argument("--port", type=int, default=8767)
    neural_serve.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    neural_serve.add_argument("--candidate-limit", type=int, default=50)
    neural_serve.add_argument("--recall-limit", type=int, default=10)
    neural_serve.add_argument("--deadline-seconds", type=float, default=2.0)
    neural_serve.add_argument(
        "--cue-mode", choices=["lexical", "semantic", "hybrid"], default="lexical"
    )
    neural_serve.add_argument("--neural-weight", type=float, default=0.05)
    neural_serve.add_argument("--neural-margin-min", type=float, default=0.0)
    neural_serve.add_argument("--neural-activation-min", type=float, default=0.7)
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
    commands.add_parser("cognitive-status")
    backfill = commands.add_parser("cognitive-backfill")
    backfill.add_argument("--max-memories", type=int, default=5000)
    timeline = commands.add_parser("timeline")
    timeline.add_argument("--limit", type=int, default=100)
    context = commands.add_parser("context")
    context.add_argument("--memory-id", type=int)
    context.add_argument("--cue", default="")
    context.add_argument("--limit", type=int, default=20)
    reactivate = commands.add_parser("reactivate")
    reactivate.add_argument("--memory-id", type=int, required=True)
    reactivate.add_argument("--cue", required=True)
    reactivate.add_argument("--prediction-error", type=float, default=0.0)
    reactivate.add_argument("--retrieval-seconds", type=float, default=0.0)
    evaluate = commands.add_parser("evaluate")
    evaluate_commands = evaluate.add_subparsers(dest="evaluate_command", required=True)
    evaluate_commands.add_parser("doctor")
    evaluate_run = evaluate_commands.add_parser("run")
    evaluate_run.add_argument(
        "--profile",
        choices=["ci", "development", "holdout", "publication"],
        default="ci",
    )
    evaluate_run.add_argument("--output", type=Path, required=True)
    evaluate_run.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    evaluate_run.add_argument("--agent-results", type=Path)
    evaluate_run.add_argument("--private-summary", type=Path)
    evaluate_run.add_argument("--neural-weight", type=float, default=0.05)
    evaluate_run.add_argument("--neural-margin-min", type=float, default=0.0)
    evaluate_run.add_argument("--neural-activation-min", type=float, default=0.0)
    agent_run = evaluate_commands.add_parser("agent-run")
    agent_run.add_argument("--runner-config", type=Path, required=True)
    agent_run.add_argument("--output", type=Path, required=True)
    agent_run.add_argument("--model-label", required=True)
    agent_run.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    agent_run.add_argument("--repeats", type=int, default=5)
    agent_run.add_argument("--scenario-limit", type=int, default=20)
    agent_run.add_argument(
        "--neural-profile", choices=["tiny", "production"], default="production"
    )
    agent_run.add_argument("--append", action="store_true")
    private_aggregate = evaluate_commands.add_parser("private-aggregate")
    private_aggregate.add_argument("--input", type=Path, required=True)
    private_aggregate.add_argument("--output", type=Path, required=True)
    private_aggregate.add_argument("--minimum-cell-size", type=int, default=10)
    evaluate_report = evaluate_commands.add_parser("report")
    evaluate_report.add_argument("run_directory", type=Path)
    evaluate_verify = evaluate_commands.add_parser("verify")
    evaluate_verify.add_argument("run_directory", type=Path)
    hermes = commands.add_parser("hermes")
    hermes_commands = hermes.add_subparsers(dest="hermes_command", required=True)
    hermes_commands.add_parser("doctor")
    for command in ("install", "uninstall"):
        action = hermes_commands.add_parser(command)
        action.add_argument("--apply", action="store_true")
    neural_migrate = commands.add_parser("neural-migrate")
    neural_migrate.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    neural_migrate.add_argument("--limit", type=int, default=100)
    neural_migrate.add_argument("--apply", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "cognitive-memory",
) -> int:
    args = _parser(prog=prog).parse_args(argv)
    if args.command == "evaluate":
        if args.evaluate_command == "doctor":
            result = evaluation_doctor()
        elif args.evaluate_command == "run":
            result = run_evaluation(
                args.output,
                profile=args.profile,
                conditions=args.conditions,
                agent_results=args.agent_results,
                private_summary=args.private_summary,
                neural_weight=args.neural_weight,
                neural_margin_min=args.neural_margin_min,
                neural_activation_min=args.neural_activation_min,
            )
        elif args.evaluate_command == "agent-run":
            result = run_agent_trials(
                args.runner_config,
                args.output,
                model_label=args.model_label,
                conditions=args.conditions,
                repeats=args.repeats,
                scenario_limit=args.scenario_limit,
                production_neural=args.neural_profile == "production",
                append=args.append,
            )
        elif args.evaluate_command == "private-aggregate":
            result = aggregate_private_results(
                args.input,
                args.output,
                minimum_cell_size=args.minimum_cell_size,
            )
        elif args.evaluate_command == "report":
            result = report_evaluation(args.run_directory)
        else:
            result = verify_evaluation(args.run_directory)
        print(json.dumps(result, indent=2, default=str))
        return 1 if result.get("status") in {"failed", "blocked"} else 0
    if args.command == "hermes":
        home = args.home or Path.home() / ".hermes"
        if args.hermes_command == "doctor":
            result = hermes_doctor(home)
        elif args.hermes_command == "install":
            result = hermes_install(home, apply=args.apply)
        else:
            result = hermes_uninstall(home, apply=args.apply)
        print(json.dumps(result, indent=2, default=str))
        return 0
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
                neuron_connectivity=circuit.neuron_connectivity,
            )
            serve(app, port=args.port)
            result = {"status": "stopped"}
        elif args.command == "neural-serve":
            token = observatory_token(
                engine.store.db_path.parent / "runtime" / "neural-readout.token",
                create=True,
            )
            assert token is not None
            runtime = NeuralReadoutRuntime(
                engine.store,
                ReadoutConfig(
                    mode="neural",
                    candidate_limit=args.candidate_limit,
                    recall_limit=args.recall_limit,
                    deadline_seconds=args.deadline_seconds,
                    cue_mode=args.cue_mode,
                    neural_weight=args.neural_weight,
                    neural_margin_min=args.neural_margin_min,
                    neural_activation_min=args.neural_activation_min,
                ),
                device=args.device,
            )
            app = create_neural_readout_app(runtime, token=token)
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
                result = synchronizer.apply(
                    plan[: args.limit], max_mutations=args.limit
                )
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
                "time_cells": len(probe["time_cell_neurons"]),
            }
        elif args.command == "neural-migrate":
            migrator = EngramMigrator(engine.store, device=args.device)
            plan = migrator.plan(args.limit)
            result = (
                migrator.apply(args.limit)
                if args.apply
                else {"status": "dry-run", "count": len(plan), "memories": plan}
            )
        elif args.command in {
            "cognitive-status",
            "cognitive-backfill",
            "timeline",
            "context",
            "reactivate",
        }:
            cognition = CognitiveMemorySystem(
                engine.store, coordinator=MemoryCoordinator(engine.store)
            )
            if args.command == "cognitive-status":
                result = cognition.status()
            elif args.command == "cognitive-backfill":
                result = cognition.backfill_existing(max_memories=args.max_memories)
            elif args.command == "timeline":
                result = {"memories": cognition.autobiographical_timeline(args.limit)}
            elif args.command == "context":
                result = cognition.reinstate_context(
                    memory_id=args.memory_id, cue=args.cue, limit=args.limit
                )
            else:
                result = cognition.reactivate(
                    args.memory_id,
                    cue=args.cue,
                    prediction_error=args.prediction_error,
                    retrieval_duration_seconds=args.retrieval_seconds,
                )
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


def legacy_main(argv: list[str] | None = None) -> int:
    """Compatibility entry point for the pre-rebrand command."""
    warnings.warn(
        "'hippocampal-memory' is deprecated; use 'cognitive-memory'.",
        DeprecationWarning,
        stacklevel=2,
    )
    return main(argv, prog="hippocampal-memory")


if __name__ == "__main__":
    raise SystemExit(main())
