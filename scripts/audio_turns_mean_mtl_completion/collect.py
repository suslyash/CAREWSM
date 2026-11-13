#!/usr/bin/env python3
"""Collect selected validation rows and descriptive TEST rows."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path("logs/audio_turns_combined_mtl_completion")
METHODS = ("progress", "ra_stch", "gradnorm", "cagrad")
COMPONENTS = ("care_depression", "care_parkinson", "wsm_depression", "wsm_parkinson")
METRICS = ("macro_f1", "weighted_f1", "macro_recall", "weighted_recall", "patient_recall", "healthy_recall")


def main() -> None:
    ledger = list(csv.DictReader((ROOT / "reports/run_ledger.csv").open(newline="", encoding="utf-8")))
    rows = []
    for item in ledger:
        if item["status"] != "COMPLETED":
            continue
        runtime = Path(item["runtime"])
        checkpoints = sorted((runtime / "checkpoints").glob("epoch=*.pt"))
        if len(checkpoints) != 1:
            raise RuntimeError(f"expected one selected checkpoint: {runtime}")
        epoch = checkpoints[0].name.split("epoch=", 1)[1].split("_", 1)[0]
        with (runtime / "summary.txt").open(newline="", encoding="utf-8") as handle:
            summary = [row for row in csv.DictReader(handle, delimiter=";") if row["epoch"] == epoch]
        if len(summary) != 1:
            raise RuntimeError(f"selected epoch not unique: {runtime}")
        source = summary[0]
        row = {"experiment_id": item["experiment_id"], "method": item["method"], "fold": item["fold"],
               "seed": "5", "selected_epoch": epoch, "checkpoint_path": str(checkpoints[0]),
               "runtime": str(runtime), "source_commit": item["source_commit"],
               "config_sha256": hashlib.sha256(Path(item["config_path"]).read_bytes()).hexdigest()}
        for split in ("val", "test"):
            for component in COMPONENTS:
                for metric in METRICS:
                    row[f"{split}_{component}_{metric}"] = source[f"{split}_{component}_{metric}"]
        row["val_overall_uar"] = str(statistics.mean(float(row[f"val_{c}_macro_recall"]) for c in COMPONENTS))
        row["test_overall_uar"] = str(statistics.mean(float(row[f"test_{c}_macro_recall"]) for c in COMPONENTS))
        rows.append(row)
    rows.sort(key=lambda row: (row["method"], int(row["fold"])))
    out = ROOT / "reports"; out.mkdir(parents=True, exist_ok=True)
    with (out / "all_results_by_fold.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    aggregates = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        if len(group) != 5: raise RuntimeError(f"{method}: expected five folds")
        aggregate = {"method": method, "completed_folds": "5"}
        for split in ("val", "test"):
            for metric in ("overall_uar",) + tuple(f"{c}_{m}" for c in COMPONENTS for m in METRICS):
                values = [float(row[f"{split}_{metric}"]) for row in group]
                aggregate[f"{split}_{metric}_mean"] = statistics.mean(values)
                aggregate[f"{split}_{metric}_sample_sd"] = statistics.stdev(values)
        aggregates.append(aggregate)
    with (out / "aggregate_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0])); writer.writeheader(); writer.writerows(aggregates)
    (out / "collection_status.json").write_text(json.dumps({"successful_slots": len(rows), "expected_slots": 20}, indent=2) + "\n")


if __name__ == "__main__": main()
