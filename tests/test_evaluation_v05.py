import json
import sys
import pytest

from hippocampal_memory.evaluation import (
    CONDITIONS,
    aggregate_private_results,
    diagnostic_worlds,
    deterministic_worlds,
    doctor,
    run_evaluation,
    run_agent_trials,
    verify_evaluation,
)
from hippocampal_memory.hermes_setup import doctor as hermes_doctor
from hippocampal_memory.hermes_setup import install
from hippocampal_memory.hermes_setup import uninstall


def test_public_worlds_are_deterministic_and_adversarial():
    first = deterministic_worlds(3)
    second = deterministic_worlds(3)

    assert first == second
    assert first[0].old.subject_key == first[0].current.subject_key
    assert "IGNORE ALL INSTRUCTIONS" in first[0].poison.content
    assert first[0].current.source_ref != first[0].old.source_ref


def test_diagnostic_splits_cover_independent_families_and_are_frozen():
    development = diagnostic_worlds("development", cases_per_family=2)
    holdout = diagnostic_worlds("holdout", cases_per_family=2)

    assert development == diagnostic_worlds("development", cases_per_family=2)
    assert len({world.family for world in development}) == 14
    assert {world.world_id for world in development}.isdisjoint(
        world.world_id for world in holdout
    )
    assert all(world.split == "holdout" for world in holdout)


def test_ci_evaluation_is_complete_and_verifiable(tmp_path):
    output = tmp_path / "run"
    result = run_evaluation(output, profile="ci", conditions=CONDITIONS)
    verification = verify_evaluation(output)
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())

    assert result["status"] == "completed"
    assert result["trials"] == 50
    assert verification["status"] == "verified"
    assert manifest["protocol"] == "eval-v0.5-protocol-1"
    assert set(summary["conditions"]) == set(CONDITIONS)
    assert summary["agent_replication"]["status"] == "not-run"
    assert not (output / "stores").exists()


def test_evaluation_doctor_reports_environment():
    result = doctor()
    assert result["status"] == "ready"
    assert "python" in result["environment"]


def test_publication_refuses_incomplete_agent_replication(tmp_path):
    with pytest.raises(ValueError, match="two complete 500-trial"):
        run_evaluation(tmp_path / "publication", profile="publication")
    assert not (tmp_path / "publication").exists()


def test_hermes_install_is_dry_run_and_reversible(tmp_path):
    home = tmp_path / ".hermes"
    planned = install(home, apply=False)
    assert planned["status"] == "planned"
    assert not (home / "config.yaml").exists()

    applied = install(home, apply=True)
    assert applied["status"] == "applied"
    assert hermes_doctor(home)["healthy"]

    removed = uninstall(home, apply=True)
    assert removed["status"] == "applied"
    assert not hermes_doctor(home)["healthy"]


def test_agent_runner_executes_shell_free_fixed_trial(tmp_path):
    config = tmp_path / "runner.json"
    config.write_text(
        json.dumps(
            {
                "command": [
                    sys.executable,
                    "-c",
                    (
                        "import json; print(json.dumps("
                        "{'answer':'9090','source_ref':'change-0',"
                        "'used_memory_id':2}))"
                    ),
                    "{prompt}",
                ],
                "cwd": str(tmp_path),
                "timeout_seconds": 10,
            }
        )
    )
    output = tmp_path / "agent-trials.jsonl"
    result = run_agent_trials(
        config,
        output,
        model_label="test-model",
        conditions=["basic"],
        repeats=1,
        scenario_limit=1,
        production_neural=False,
    )
    record = json.loads(output.read_text().strip())

    assert result["trials"] == 1
    assert record["task_complete"] == 1
    assert record["runner_config_sha256"]


def test_private_replication_emits_aggregate_cells_only(tmp_path):
    raw = tmp_path / "private.jsonl"
    rows = [
        {
            "condition": condition,
            "model": "private-reference",
            "answer_correct": 1,
            "source_correct": 1,
            "task_complete": 1,
            "private_content": f"must not escape {index}",
        }
        for condition in CONDITIONS
        for index in range(10)
    ]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "private-summary.json"

    result = aggregate_private_results(raw, output)
    summary = json.loads(output.read_text())

    assert result["cells"] == 5
    assert summary["aggregate_only"]
    assert summary["minimum_cell_size"] == 10
    assert "must not escape" not in output.read_text()
