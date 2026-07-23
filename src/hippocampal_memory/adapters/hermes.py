"""Thin adapter for a Hermes installation."""

from __future__ import annotations

import os

from ..replay import DEFAULT_MODEL, HippocampusEngine, ReplayConfig


def create_engine() -> HippocampusEngine:
    """Create an engine using the active Hermes home and plugin settings."""
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    os.environ.setdefault("HIPPOCAMPAL_MEMORY_HOME", str(home))
    values = {}
    config_path = home / "config.yaml"
    if config_path.exists():
        try:
            import yaml

            root = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
            values = root.get("plugins", {}).get("hermes-memory-store", {})
        except Exception:
            values = {}
    enabled = str(values.get("hippocampus_enabled", "true")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return HippocampusEngine(
        home=home,
        state_db=home / "state.db",
        config=ReplayConfig(
            enabled=enabled,
            model=str(values.get("hippocampus_model", DEFAULT_MODEL)),
            hrr_dim=int(values.get("hrr_dim", 4096)),
        ),
    )
