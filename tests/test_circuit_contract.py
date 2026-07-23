from dataclasses import replace

import pytest

from hippocampal_memory.circuit import CircuitConfig


def test_production_circuit_has_actual_36k_neurons() -> None:
    config = CircuitConfig()
    config.validate()
    assert sum(config.populations.values()) == 36_864
    assert config.populations["DG"] > config.populations["EC"]
    assert {"EC_DG", "DG_CA3", "CA3_CA1"} <= set(config.fanout)


def test_circuit_bounds_fail_closed() -> None:
    with pytest.raises(ValueError):
        replace(
            CircuitConfig(),
            populations={"EC": 100_000, "DG": 100_000, "CA3": 100_000, "CA1": 1},
        ).validate()
