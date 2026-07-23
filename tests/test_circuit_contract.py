from dataclasses import replace

import pytest

from hippocampal_memory.circuit import CircuitConfig, TrisynapticCircuit


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


def test_time_cells_encode_sequence_and_context() -> None:
    pytest.importorskip("torch", reason="neural extra is optional")
    config = replace(
        CircuitConfig(),
        populations={"EC": 128, "DG": 256, "CA3": 128, "CA1": 64},
        fanout={
            "EC_DG": 2,
            "DG_CA3": 2,
            "EC_CA3": 2,
            "CA3_CA3": 2,
            "CA3_CA1": 2,
            "EC_CA1": 2,
            "LOCAL_INHIBITION": 2,
        },
    )
    circuit = TrisynapticCircuit(config, device="cpu")

    first = circuit.temporal_current(0, 20, context_key="event:a")
    middle = circuit.temporal_current(10, 20, context_key="event:a")
    remapped = circuit.temporal_current(0, 20, context_key="event:b")

    assert circuit.time_cell_ids.numel() >= 32
    assert not first.equal(middle)
    assert not first.equal(remapped)
    result = circuit.stimulate_engram("memory:1", steps=20, plastic=False)
    assert result["status"] == "completed"
    assert result["time_cell_neurons"]
    assert any(frame["time_cells_active"] for frame in result["frames"])
    geometry = circuit.geometry()
    assert geometry["time_cells"]["mechanism"].startswith("context-remapped")
