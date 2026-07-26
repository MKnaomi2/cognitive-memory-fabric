"""Private, local-only effectiveness evaluation for narrative recall."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from .narrative import NarrativeEngine
from .store import MemoryStore

FAMILIES = {
    "exact": 25,
    "partial": 25,
    "paraphrase": 40,
    "associative": 30,
    "temporal": 30,
    "event-timeline": 50,
    "project-evolution": 40,
    "thematic": 35,
    "source-attribution": 25,
}


def _tokens(value: str) -> list[str]:
    return [
        token.strip(".,:;!?()[]{}\"'").lower()
        for token in value.split()
        if len(token.strip(".,:;!?()[]{}\"'")) >= 4
    ]


def _cue(content: str, family: str) -> str:
    tokens = _tokens(content)
    if not tokens:
        return content[:120]
    if family == "exact":
        return " ".join(tokens[:6])
    if family == "partial":
        return " ".join(tokens[::2][:8])
    if family == "paraphrase":
        return "What do I remember concerning " + " ".join(tokens[-6:])
    return " ".join(tokens[:8])


def _stable(rows: list[Any], family: str) -> list[Any]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{family}:{int(row['fact_id'])}".encode()
        ).hexdigest(),
    )


def _cases(store: MemoryStore, limit: int) -> list[dict[str, Any]]:
    rows = store._conn.execute(
        """
        SELECT f.* FROM facts f JOIN engram_bindings b
          ON b.memory_id=CAST(f.fact_id AS TEXT)
        WHERE f.status='active' AND b.encoding_version='content-v3'
          AND COALESCE(b.ca1_signature_json,'[]')!='[]'
        ORDER BY f.fact_id
        """
    ).fetchall()
    if not rows:
        raise RuntimeError("no current neural memories are available")
    contexts: dict[str, list[Any]] = {}
    for row in rows:
        context = str(row["context_id"] or "")
        if context:
            contexts.setdefault(context, []).append(row)
    cases: list[dict[str, Any]] = []
    single = {"exact", "partial", "paraphrase", "source-attribution"}
    for family, requested in FAMILIES.items():
        requested = min(requested, max(0, limit - len(cases)))
        if requested <= 0:
            break
        if family in single:
            for row in _stable(rows, family)[:requested]:
                cases.append(
                    {
                        "case_id": f"{family}-{row['fact_id']}",
                        "family": family,
                        "query": _cue(str(row["content"]), family),
                        "gold_memory_ids": [int(row["fact_id"])],
                    }
                )
            continue
        groups = [
            sorted(
                values,
                key=lambda item: (
                    int(item["sequence_index"] or 0),
                    int(item["fact_id"]),
                ),
            )
            for values in contexts.values()
            if len(values) >= 3
        ]
        groups.sort(
            key=lambda values: hashlib.sha256(
                f"{family}:{values[0]['context_id']}".encode()
            ).hexdigest()
        )
        for index in range(requested):
            group = groups[index % len(groups)] if groups else _stable(rows, family)[:3]
            anchor = group[index % len(group)]
            cases.append(
                {
                    "case_id": f"{family}-{index:04d}",
                    "family": family,
                    "query": _cue(str(anchor["content"]), family),
                    "gold_memory_ids": [int(row["fact_id"]) for row in group[:6]],
                }
            )
    return cases[:limit]


def run_narrative_evaluation(
    database: str | Path,
    output: str | Path,
    *,
    cases: int = 300,
    split: str = "development",
) -> dict[str, Any]:
    if split not in {"development", "validation"}:
        raise ValueError("split must be development or validation")
    requested = max(10, min(1000, int(cases)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    store = MemoryStore(database)
    try:
        generated = _cases(store, requested)
        selected = [
            case
            for case in generated
            if (
                int(hashlib.sha256(case["case_id"].encode()).hexdigest()[:8], 16)
                % 10
                < 7
            )
            == (split == "development")
        ]
        engine = NarrativeEngine(store)
        results = []
        for case in selected:
            for condition, include_neural in (
                ("evidence-only", False),
                ("evidence-plus-neural", True),
            ):
                narrative = engine.compose(
                    case["query"], include_neural=include_neural
                )
                selected_ids = {
                    int(memory["memory_id"]) for memory in narrative["memories"]
                }
                gold = set(case["gold_memory_ids"])
                cited = {
                    int(source_id)
                    for claim in narrative["claims"]
                    for source_id in claim["source_ids"]
                }
                results.append(
                    {
                        **case,
                        "condition": condition,
                        "retrieved_ids": sorted(selected_ids),
                        "source_coverage": len(selected_ids & gold) / len(gold),
                        "citation_precision": (
                            len(cited & selected_ids) / len(cited) if cited else 1.0
                        ),
                        "unsupported_claims": sum(
                            not claim["source_ids"] for claim in narrative["claims"]
                        ),
                        "unlabeled_inferences": sum(
                            claim["relation"]
                            in {"same_context", "neural_association"}
                            and claim["kind"] != "inference"
                            for claim in narrative["claims"]
                        ),
                        "structure": narrative["structure"],
                    }
                )
        by_condition = {
            condition: [row for row in results if row["condition"] == condition]
            for condition in ("evidence-only", "evidence-plus-neural")
        }
        condition_metrics = {
            condition: {
                "source_coverage_mean": statistics.mean(
                    row["source_coverage"] for row in rows
                )
                if rows
                else 0.0,
                "citation_precision_mean": statistics.mean(
                    row["citation_precision"] for row in rows
                )
                if rows
                else 0.0,
                "unsupported_claims": sum(row["unsupported_claims"] for row in rows),
                "unlabeled_inferences": sum(
                    row["unlabeled_inferences"] for row in rows
                ),
            }
            for condition, rows in by_condition.items()
        }
        neural_metrics = condition_metrics["evidence-plus-neural"]
        baseline_metrics = condition_metrics["evidence-only"]
        summary = {
            "status": "completed",
            "profile": f"private-narrative-{split}",
            "cases": len(selected),
            "trials": len(results),
            "source_coverage_mean": neural_metrics["source_coverage_mean"],
            "citation_precision_mean": neural_metrics["citation_precision_mean"],
            "unsupported_claims": neural_metrics["unsupported_claims"],
            "unlabeled_inferences": neural_metrics["unlabeled_inferences"],
            "condition_metrics": condition_metrics,
            "neural_source_coverage_delta": (
                neural_metrics["source_coverage_mean"]
                - baseline_metrics["source_coverage_mean"]
            ),
            "family_counts": {
                family: sum(case["family"] == family for case in selected)
                for family in FAMILIES
            },
            "privacy": "raw cases are local-only and must not be committed",
        }
        trials = output / "trials.jsonl"
        trials.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in results),
            encoding="utf-8",
        )
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        manifest = {
            "profile": summary["profile"],
            "database_sha256": hashlib.sha256(Path(database).read_bytes()).hexdigest(),
            "artifacts": {
                "trials.jsonl": hashlib.sha256(trials.read_bytes()).hexdigest(),
                "summary.json": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            },
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return summary
    finally:
        store.close()
