#!/usr/bin/env python3
"""Collect selected VAL and descriptive TEST metrics for the 25-run matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

from mlflow.tracking import MlflowClient

ROOT = Path("logs/audio_interpretable_feature_fusion")
COMBINATIONS = ("B", "BE", "BP", "BA", "BEPA")
COMPONENTS = ("care_depression", "care_parkinson", "wsm_depression", "wsm_parkinson")
METRICS = (
    "macro_f1",
    "weighted_f1",
    "macro_recall",
    "weighted_recall",
    "patient_recall",
    "healthy_recall",
)


def selected_row(runtime: Path) -> tuple[str, dict[str, str], Path]:
    checkpoints = sorted((runtime / "checkpoints").glob("epoch=*.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one selected checkpoint: {runtime}")
    epoch = checkpoints[0].name.split("epoch=", 1)[1].split("_", 1)[0]
    with (runtime / "summary.txt").open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter=";")
            if row["epoch"] == epoch
        ]
    if len(rows) != 1:
        raise RuntimeError(f"selected epoch is not unique: {runtime}")
    return epoch, rows[0], checkpoints[0]


def mlflow_run_map() -> dict[str, object]:
    client = MlflowClient("sqlite:///logs/audio_interpretable_feature_fusion/mlflow.db")
    experiment = client.get_experiment_by_name("audio_interpretable_feature_fusion")
    if experiment is None:
        raise RuntimeError("MLflow experiment is missing")
    runs = client.search_runs([experiment.experiment_id], max_results=1000)
    return {run.data.tags.get("mlflow.runName"): run for run in runs}


def main() -> None:
    run_map = mlflow_run_map()
    ledger_path = ROOT / "reports/run_ledger.csv"
    ledger = list(csv.DictReader(ledger_path.open(newline="", encoding="utf-8")))
    rows = []
    provenance = []
    for item in ledger:
        if item["status"] != "COMPLETED":
            continue
        epoch, source, checkpoint = selected_row(Path(item["runtime"]))
        run = run_map.get(Path(item["runtime"]).name)
        if run is None:
            raise RuntimeError(f"missing MLflow run for {item['experiment_id']}")
        row = {
            "experiment_id": item["experiment_id"],
            "combination": item["combination"],
            "fold": item["fold"],
            "seed": "5",
            "selected_epoch": epoch,
            "checkpoint_path": str(checkpoint),
            "runtime": item["runtime"],
            "source_commit": item["source_commit"],
            "config_sha256": hashlib.sha256(
                Path(item["config_path"]).read_bytes()
            ).hexdigest(),
        }
        for split in ("val", "test"):
            for component in COMPONENTS:
                for metric in METRICS:
                    row[f"{split}_{component}_{metric}"] = source[
                        f"{split}_{component}_{metric}"
                    ]
        row["val_overall_macro_recall"] = str(
            statistics.mean(
                float(row[f"val_{component}_macro_recall"]) for component in COMPONENTS
            )
        )
        row["test_overall_macro_recall"] = str(
            statistics.mean(
                float(row[f"test_{component}_macro_recall"]) for component in COMPONENTS
            )
        )
        provenance.append(
            {
                "experiment_id": item["experiment_id"],
                "combination": item["combination"],
                "fold": item["fold"],
                "mlflow_run_id": run.info.run_id,
                "mlflow_status": run.info.status,
                "runtime_evidence_status": "verified_train_log_summary_selected_checkpoint",
                "selected_epoch": epoch,
                "selected_val_overall_macro_recall": row["val_overall_macro_recall"],
                "checkpoint_path": str(checkpoint),
                "runtime": item["runtime"],
                "source_commit": item["source_commit"],
                "config_sha256": row["config_sha256"],
            }
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (COMBINATIONS.index(row["combination"]), int(row["fold"]))
    )
    out = ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out / "all_results_by_fold.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    aggregates = []
    aggregate_metrics = ("overall_macro_recall",) + tuple(
        f"{component}_{metric}" for component in COMPONENTS for metric in METRICS
    )
    for combination in COMBINATIONS:
        group = [row for row in rows if row["combination"] == combination]
        if len(group) != 5:
            continue
        aggregate = {"combination": combination, "completed_folds": "5"}
        for split in ("val", "test"):
            for metric in aggregate_metrics:
                values = [float(row[f"{split}_{metric}"]) for row in group]
                aggregate[f"{split}_{metric}_mean"] = statistics.mean(values)
                aggregate[f"{split}_{metric}_sample_sd"] = statistics.stdev(values)
        aggregates.append(aggregate)
    if aggregates:
        with (out / "aggregate_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
            writer.writeheader()
            writer.writerows(aggregates)
    provenance.sort(
        key=lambda row: (COMBINATIONS.index(row["combination"]), int(row["fold"]))
    )
    if provenance:
        with (out / "provenance_ledger.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(provenance[0]))
            writer.writeheader()
            writer.writerows(provenance)
    status = {
        "successful_slots": len(rows),
        "expected_slots": 25,
        "completed_combinations": [row["combination"] for row in aggregates],
    }
    (out / "collection_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
