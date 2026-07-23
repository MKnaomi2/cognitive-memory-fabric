"""Command-line controls for replay and lifecycle maintenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .replay import DEFAULT_MODEL, DEFAULT_URL, HippocampusEngine, ReplayConfig


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
        else:
            result = engine.store.run_forgetting_maintenance()
        print(json.dumps(result, indent=2, default=str))
        return 1 if result.get("status") == "failed" else 0
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
