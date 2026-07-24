"""Reversible Hermes provider configuration helpers."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _yaml() -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install the 'hermes' extra for YAML configuration") from exc
    return yaml


def doctor(home: Path) -> dict[str, Any]:
    config_path = home / "config.yaml"
    plugin_path = home / "plugins" / "cognitive-memory-fabric"
    result: dict[str, Any] = {
        "home": str(home),
        "config": str(config_path),
        "config_exists": config_path.exists(),
        "provider": None,
        "plugin": str(plugin_path),
        "plugin_exists": (plugin_path / "__init__.py").exists(),
        "healthy": False,
    }
    if not config_path.exists():
        result["issues"] = ["Hermes config.yaml was not found"]
        return result
    root = _yaml().safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    memory = root.get("memory", {})
    result["provider"] = memory.get("provider")
    result["healthy"] = (
        result["provider"] == "cognitive-memory-fabric"
        and result["plugin_exists"]
    )
    result["issues"] = []
    if result["provider"] != "cognitive-memory-fabric":
        result["issues"].append("provider is not selected")
    if not result["plugin_exists"]:
        result["issues"].append("provider plugin is not installed")
    return result


def install(home: Path, *, apply: bool = False) -> dict[str, Any]:
    config_path = home / "config.yaml"
    source_plugin = Path(__file__).resolve().parents[2] / "integrations" / "hermes"
    plugin_path = home / "plugins" / "cognitive-memory-fabric"
    root = (
        _yaml().safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
        if config_path.exists()
        else {}
    )
    before = json.loads(json.dumps(root))
    memory = root.setdefault("memory", {})
    memory.update(
        {
            "provider": "cognitive-memory-fabric",
            "replay_mode": "none",
            "candidate_limit": 50,
            "recall_limit": 10,
            "max_injected_chars": 8000,
            "deadline_seconds": 2.0,
        }
    )
    result = {
        "status": "planned",
        "before": before.get("memory"),
        "after": memory,
        "plugin_source": str(source_plugin),
        "plugin_target": str(plugin_path),
    }
    if not apply:
        return result
    home.mkdir(parents=True, exist_ok=True)
    backup = None
    if config_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = config_path.with_name(f"config.yaml.cmf-{stamp}.bak")
        shutil.copy2(config_path, backup)
    plugin_backup = None
    if plugin_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        plugin_backup = plugin_path.with_name(
            f"cognitive-memory-fabric.cmf-{stamp}.bak"
        )
        plugin_path.replace(plugin_backup)
    plugin_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_plugin, plugin_path)
    temporary = config_path.with_suffix(".yaml.tmp")
    temporary.write_text(
        _yaml().safe_dump(root, sort_keys=False), encoding="utf-8", newline="\n"
    )
    temporary.replace(config_path)
    result.update(
        status="applied",
        backup=str(backup) if backup else None,
        plugin_backup=str(plugin_backup) if plugin_backup else None,
    )
    return result


def uninstall(home: Path, *, apply: bool = False) -> dict[str, Any]:
    config_path = home / "config.yaml"
    plugin_path = home / "plugins" / "cognitive-memory-fabric"
    if not config_path.exists():
        return {"status": "unchanged", "reason": "config does not exist"}
    root = _yaml().safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    memory = root.get("memory", {})
    if memory.get("provider") != "cognitive-memory-fabric":
        return {"status": "unchanged", "reason": "provider is not selected"}
    result = {
        "status": "planned",
        "removed": dict(memory),
        "plugin": str(plugin_path),
    }
    if not apply:
        return result
    backup = config_path.with_suffix(".yaml.cmf-uninstall.bak")
    shutil.copy2(config_path, backup)
    root.pop("memory", None)
    config_path.write_text(
        _yaml().safe_dump(root, sort_keys=False), encoding="utf-8", newline="\n"
    )
    plugin_backup = None
    if plugin_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        plugin_backup = plugin_path.with_name(
            f"cognitive-memory-fabric.cmf-uninstalled-{stamp}"
        )
        plugin_path.replace(plugin_backup)
    result.update(
        status="applied",
        backup=str(backup),
        plugin_backup=str(plugin_backup) if plugin_backup else None,
    )
    return result
