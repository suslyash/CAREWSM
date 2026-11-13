#!/usr/bin/env python3
"""Validate the committed CAREWSM audio research evidence tables."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "logs/audio_turns_combined/reports"
ABLATION = ROOT / "logs/audio_turns_combined_ablation/reports"
TOLERANCE = 1e-4

REQUIRED_METRICS = (
    "macro_f1",
    "weighted_f1",
    "macro_recall",
    "weighted_recall",
    "patient_recall",
    "healthy_recall",
)
GROUPS = (
    "care_depression",
    "care_parkinson",
    "wsm_depression",
    "wsm_parkinson",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_finite(row: dict[str, str], fields: list[str], identity: str) -> None:
    for field in fields:
        require(field in row and row[field] != "", f"{identity}: missing {field}")
        value = float(row[field])
        require(math.isfinite(value), f"{identity}: non-finite {field}")


def check_metrics(rows: list[dict[str, str]], label: str) -> None:
    identities = {(row["experiment_id"], row["fold"]) for row in rows}
    require(len(identities) == len(rows), f"{label}: duplicate experiment/fold identity")
    metric_fields = [
        f"{prefix}_{group}_{metric}"
        for prefix in ("val", "test")
        for group in GROUPS
        for metric in REQUIRED_METRICS
    ]
    for row in rows:
        identity = f"{label}:{row['experiment_id']}/fold{row['fold']}"
        require(row["status"] == "COMPLETED", f"{identity}: incomplete status")
        check_finite(row, metric_fields + ["val_overall_uar", "test_overall_uar"], identity)
        val_components = [float(row[f"val_{group}_macro_recall"]) for group in GROUPS]
        test_components = [float(row[f"test_{group}_macro_recall"]) for group in GROUPS]
        require(
            math.isclose(float(row["val_overall_uar"]), statistics.mean(val_components), abs_tol=TOLERANCE),
            f"{identity}: VAL UAR does not equal four-component mean",
        )
        require(
            math.isclose(float(row["test_overall_uar"]), statistics.mean(test_components), abs_tol=TOLERANCE),
            f"{identity}: TEST UAR does not equal four-component mean",
        )


def check_provenance(rows: list[dict[str, str]], ledger: list[dict[str, str]], label: str) -> None:
    by_identity = {(row["experiment_id"], row["fold"]): row for row in rows}
    ledger_identity = {(row["experiment_id"], row["fold"]): row for row in ledger}
    require(by_identity.keys() == ledger_identity.keys(), f"{label}: ledger identity mismatch")
    for identity, row in ledger_identity.items():
        name = f"{label}:{identity[0]}/fold{identity[1]}"
        for field in ("source_commit", "config_path", "runtime_directory", "mlflow_run_id", "selected_epoch"):
            require(row.get(field, "") != "", f"{name}: missing provenance field {field}")
        require(row.get("status") == "COMPLETED", f"{name}: ledger is not completed")
        require(row.get("evidence_ok") == "YES", f"{name}: evidence_ok is not YES")
        require(row["selected_epoch"].isdigit(), f"{name}: invalid selected epoch")


def check_aggregate(rows: list[dict[str, str]], summary: list[dict[str, str]], label: str) -> None:
    require(len(summary) == len({row["experiment_id"] for row in rows}), f"{label}: summary row count mismatch")
    for experiment_id in sorted({row["experiment_id"] for row in rows}):
        fold_rows = [row for row in rows if row["experiment_id"] == experiment_id]
        aggregate = next(row for row in summary if row["experiment_id"] == experiment_id)
        require(int(aggregate["completed_folds"]) == 5, f"{label}:{experiment_id}: not five folds")
        for prefix in ("val", "test"):
            values = [float(row[f"{prefix}_overall_uar"]) for row in fold_rows]
            mean = statistics.mean(values)
            sample_sd = statistics.stdev(values)
            require(math.isclose(float(aggregate[f"{prefix}_overall_uar_mean"]), mean, abs_tol=TOLERANCE), f"{label}:{experiment_id}: {prefix} mean mismatch")
            require(math.isclose(float(aggregate[f"{prefix}_overall_uar_sample_sd"]), sample_sd, abs_tol=TOLERANCE), f"{label}:{experiment_id}: {prefix} sample SD mismatch")


def check_config_monitors() -> None:
    paths = sorted((ROOT / "configs/audio_turns_combined_ablation/ablation").glob("AC_mean_static_gated64_fold*.yaml"))
    require(len(paths) == 5, "ablation: five exact configs are required")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        require("val_mean_corpora_tasks/macro_recall" in text, f"{path}: missing declared VAL monitor")


def main() -> int:
    try:
        main_rows = read_rows(MAIN / "all_results_by_fold.csv")
        ablation_rows = read_rows(ABLATION / "all_results_by_fold.csv")
        require(len(main_rows) == 145, "main study must contain 145 fold slots")
        require(len(ablation_rows) == 5, "ablation must contain five fold slots")
        require(len({row["experiment_id"] for row in main_rows}) == 29, "main study must contain 29 configurations")
        require({row["experiment_id"] for row in ablation_rows} == {"AC_mean_static_gated64"}, "unexpected ablation configuration")
        require(sorted({int(row["fold"]) for row in main_rows}) == [0, 1, 2, 3, 4], "main folds are not 0..4")
        require(sorted({int(row["fold"]) for row in ablation_rows}) == [0, 1, 2, 3, 4], "ablation folds are not 0..4")
        check_metrics(main_rows, "main")
        check_metrics(ablation_rows, "ablation")
        check_provenance(main_rows, read_rows(MAIN / "run_ledger.csv"), "main")
        check_provenance(ablation_rows, read_rows(ABLATION / "run_ledger.csv"), "ablation")
        check_aggregate(main_rows, read_rows(MAIN / "all_results_summary.csv"), "main")
        check_aggregate(ablation_rows, read_rows(ABLATION / "all_results_summary.csv"), "ablation")
        check_config_monitors()
    except (OSError, KeyError, ValueError, statistics.StatisticsError) as error:
        print(f"EVIDENCE VALIDATION FAILED: {error}", file=sys.stderr)
        return 1
    print("EVIDENCE VALIDATION PASSED: 29 main configurations + 1 AC/64 ablation, five folds each")
    print("This checks committed numerical evidence and provenance, not independent GPU re-execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
