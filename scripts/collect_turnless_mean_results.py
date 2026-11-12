#!/usr/bin/env python3
"""Collect selected VAL/TEST rows and generate the turnless mean report."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import yaml


EXPERIMENTS = ("E1_gated64", "E2_mlp64", "E3_gated32", "E4_mlp32")
METRICS = (
    "macro_recall",
    "patient_recall",
    "healthy_recall",
    "macro_f1",
    "weighted_f1",
    "weighted_recall",
)
COMPONENTS = (
    "care_depression",
    "care_parkinson",
    "wsm_depression",
    "wsm_parkinson",
)
BASELINE_VAL = [0.8321, 0.8400, 0.7979, 0.8010, 0.8643]
BASELINE_TEST = [0.7547, 0.7325, 0.7609, 0.7846, 0.7888]


def runtime_for(run_name: str) -> Path | None:
    root = Path("logs/audio_representation_pilot/turnless_mean")
    candidates = sorted(root.glob(f"audio_representation_pilot_turnless_mean/{run_name}_*"))
    if len(candidates) != 1:
        return None
    return candidates[0]


def selected_row(summary: Path) -> dict[str, str] | None:
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    valid = [row for row in rows if row.get("val_mean_corpora_tasks_macro_recall") not in (None, "-")]
    if not valid:
        return None
    return max(valid, key=lambda row: (float(row["val_mean_corpora_tasks_macro_recall"]), int(row["epoch"])))


def finite(value: str | None) -> float | None:
    if value in (None, "", "-"):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    fold_rows: list[dict[str, object]] = []
    ledger: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        model = "carewsm_gated_fusion" if "gated" in experiment else "carewsm_pooled_mlp"
        hidden = 64 if experiment.endswith("64") else 32
        for fold in range(5):
            run_name = f"{experiment}_fold{fold}"
            runtime = runtime_for(run_name)
            row = selected_row(runtime / "summary.txt") if runtime and (runtime / "summary.txt").is_file() else None
            status = "COMPLETED" if row and runtime and (runtime / "train.log").is_file() and "Training complete" in (runtime / "train.log").read_text(encoding="utf-8") and list((runtime / "checkpoints").glob("*.pt")) else "MISSING"
            if status == "MISSING" and not args.allow_incomplete:
                raise SystemExit(f"missing completed run: {run_name}")
            output: dict[str, object] = {
                "experiment": experiment,
                "fold": fold,
                "model": model,
                "hidden_dim": hidden,
                "pooling": "mean",
                "status": status,
                "runtime_directory": str(runtime) if runtime else "",
                "selected_epoch": row.get("epoch", "") if row else "",
                "selected_val_uar": row.get("val_mean_corpora_tasks_macro_recall", "") if row else "",
                "selected_checkpoint": "",
            }
            if runtime and row:
                checkpoints = sorted(runtime.joinpath("checkpoints").glob("*.pt"))
                matching = [p for p in checkpoints if f"epoch={row['epoch']}_" in p.name]
                output["selected_checkpoint"] = str(matching[0] if matching else checkpoints[0]) if checkpoints else ""
                for split in ("val", "test"):
                    for component in COMPONENTS:
                        for metric in METRICS:
                            output[f"{split}_{component}_{metric}"] = row.get(f"{split}_{component}_{metric}", "")
                    output[f"{split}_overall_uar"] = row.get(f"{split}_mean_corpora_tasks_macro_recall", "")
            fold_rows.append(output)
            ledger.append({
                "experiment": experiment,
                "fold": fold,
                "model": model,
                "pooling": "mean",
                "hidden_dim": hidden,
                "git_commit": "8a91a06e52d4710f54503b824f76a969cf73942b",
                "seed": 5,
                "selected_epoch": output["selected_epoch"],
                "selected_checkpoint": output["selected_checkpoint"],
                "selection_monitor": "val_mean_corpora_tasks/macro_recall",
                "runtime_directory": output["runtime_directory"],
                "mlflow_run_id": "unavailable_until_runtime_query",
                "status": status,
                "failure_reason": "" if status == "COMPLETED" else "missing summary/checkpoint",
            })

    summary_rows: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        for split in ("val", "test"):
            values = [finite(str(row.get(f"{split}_overall_uar", ""))) for row in fold_rows if row["experiment"] == experiment]
            values = [value for value in values if value is not None]
            summary_rows.append({
                "experiment": experiment,
                "split": split,
                "completed_folds": len(values),
                "mean_overall_uar": statistics.mean(values) if values else "",
                "sample_sd": statistics.stdev(values) if len(values) > 1 else "",
            })
    write_csv(Path("turnless_mean_results_by_fold.csv"), fold_rows)
    write_csv(Path("turnless_mean_results_summary.csv"), summary_rows)
    write_csv(Path("turnless_mean_run_ledger.csv"), ledger)

    complete = [row for row in fold_rows if row["status"] == "COMPLETED"]
    lines = [
        "# Turnless Mean-Pooling Five-Fold Experiment",
        "",
        "This report is generated from the selected validation checkpoint for each run.",
        "TEST values are descriptive and were not used for selection.",
        "The historical TEST split is previously observed descriptive evidence, not a pristine holdout.",
        "",
        f"Completed runs: {len(complete)} / 20",
        "",
        "## Validation and descriptive TEST overall UAR",
        "",
        "| Experiment | VAL folds | VAL mean | VAL SD | TEST folds | TEST mean | TEST SD |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for experiment in EXPERIMENTS:
        vals = [finite(str(row.get("val_overall_uar", ""))) for row in fold_rows if row["experiment"] == experiment]
        tests = [finite(str(row.get("test_overall_uar", ""))) for row in fold_rows if row["experiment"] == experiment]
        vals = [value for value in vals if value is not None]
        tests = [value for value in tests if value is not None]
        lines.append(f"| {experiment} | {', '.join(f'{v:.4f}' for v in vals)} | {statistics.mean(vals):.4f} | {statistics.stdev(vals):.4f} | {', '.join(f'{v:.4f}' for v in tests)} | {statistics.mean(tests):.4f} | {statistics.stdev(tests):.4f} |" if len(vals) > 1 and len(tests) > 1 else f"| {experiment} | incomplete | — | — | incomplete | — | — |")
    lines += [
        "",
        "## Required six-metric component records",
        "",
        "The machine-readable `turnless_mean_results_by_fold.csv` contains all six metrics for each corpus/task component on VAL and TEST: Macro F1, Weighted F1, Macro Recall, Weighted Recall, Patient Recall, and Healthy Recall.",
        "Confusion counts are unavailable from the existing wide summary callback and are marked unavailable rather than fabricated.",
        "",
        "## Data and reproducibility",
        "",
        "- Feature families: `emotionmsp_embeds` + `trillsson4_1`, ordered as listed.",
        "- Pooling: independent arithmetic mean over valid turns, then concatenation; input shape `[batch, 1, 2048]`.",
        "- Joint CARE+WSM five-fold training, seed 5, AdamW, learning rate 1e-4, weight decay 0, batch size 64, dropout 0.1, maximum 50 epochs, patience 7.",
        "- Parkinson class weights are the existing fold-specific TRAIN-only values.",
        "- Data audit: `logs/audio_representation_pilot/turnless_mean/data_verification.json`.",
        "- Run ledger: `turnless_mean_run_ledger.csv`.",
        "",
        "## Acceptance",
        "",
        "Acceptance is validation-only: at least +0.010 mean overall VAL UAR versus Weighted-F, no component mean deterioration over 0.020, and paired-fold support. Decisions remain `INCOMPLETE` until all five folds for a configuration are available; the final collector will preserve the actual result.",
    ]
    report = Path("docs/evidence/TURNLESS_MEAN_FIVEFOLD_RESULTS_EN.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Collected {len(complete)}/20 completed runs")


if __name__ == "__main__":
    main()
