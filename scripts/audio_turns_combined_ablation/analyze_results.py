#!/usr/bin/env python3
"""Reconcile the isolated AC static gated64 ablation."""

from __future__ import annotations

import csv
import re
import statistics
import subprocess
from pathlib import Path

import yaml
from mlflow.tracking import MlflowClient

ROOT = Path("logs/audio_turns_combined_ablation")
CONFIG_ROOT = Path("configs/audio_turns_combined_ablation/ablation")
RUNTIME_ROOT = ROOT / "audio_turns_combined_ablation"
COMPONENTS = ("care_depression", "care_parkinson", "wsm_depression", "wsm_parkinson")
METRICS = ("macro_f1", "weighted_f1", "macro_recall", "weighted_recall", "patient_recall", "healthy_recall")
CHECKPOINT = re.compile(r"^epoch=(\d+)_val_mean_corpora_tasks_macro_recall=([0-9.]+)\.pt$")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def load_run(config_path: Path, client: MlflowClient) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    info = config["experiment_info"]["params"]
    run_name = info["run_name"]
    runtime = sorted(RUNTIME_ROOT.glob(run_name + "_*"))
    if len(runtime) != 1:
        raise RuntimeError(f"expected one runtime for {run_name}, found {runtime}")
    runtime = runtime[0]
    checkpoints = [path for path in (runtime / "checkpoints").glob("*.pt") if path.name != "last.pt"]
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one selected checkpoint for {run_name}")
    match = CHECKPOINT.fullmatch(checkpoints[0].name)
    if match is None:
        raise RuntimeError(f"invalid checkpoint name: {checkpoints[0]}")
    with (runtime / "summary.txt").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle, delimiter=";"))
    selected = [row for row in summary if row.get("epoch") == match.group(1)]
    if len(selected) != 1:
        raise RuntimeError(f"selected epoch missing from summary: {run_name}")
    row = selected[0]
    if abs(float(row["val_mean_corpora_tasks_macro_recall"]) - float(match.group(2))) > 5e-5:
        raise RuntimeError(f"checkpoint/summary mismatch: {run_name}")
    experiment = client.get_experiment_by_name("audio_turns_combined_ablation")
    runs = [run for run in client.search_runs([experiment.experiment_id]) if run.info.run_name.startswith(run_name + "_")]
    if len(runs) != 1 or runs[0].info.status != "FINISHED":
        raise RuntimeError(f"MLflow provenance invalid: {run_name}")
    result: dict[str, object] = {
        "stage": "ablation",
        "experiment_id": "AC_mean_static_gated64",
        "feature_combination": "AC",
        "feature_families": "emotionmsp_embeds+trillsson4_1",
        "representation": "mean",
        "model": "carewsm_static_gated_fusion",
        "hidden_dim": 64,
        "fold": config["data"]["params"]["fold"],
        "seed": config["seed"],
        "status": "COMPLETED",
        "selected_epoch": int(match.group(1)),
        "selected_val_uar": float(row["val_mean_corpora_tasks_macro_recall"]),
        "checkpoint_path": str(checkpoints[0]),
        "mlflow_run_id": runs[0].info.run_id,
        "config_path": str(config_path),
        "runtime_directory": str(runtime),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "evidence_ok": "YES",
    }
    for split in ("val", "test"):
        for component in COMPONENTS:
            for metric in METRICS:
                result[f"{split}_{component}_{metric}"] = float(row[f"{split}_{component}_{metric}"])
        result[f"{split}_overall_uar"] = float(row[f"{split}_mean_corpora_tasks_macro_recall"])
    return result


def existing_summary() -> dict[str, tuple[float, float, float, float]]:
    rows = list(csv.DictReader(Path("logs/audio_turns_combined/reports/all_results_summary.csv").open(encoding="utf-8")))
    result = {}
    for row in rows:
        result[row["experiment_id"]] = (
            float(row["val_overall_uar_mean"]), float(row["val_overall_uar_sample_sd"]),
            float(row["test_overall_uar_mean"]), float(row["test_overall_uar_sample_sd"]),
        )
    return result


def main() -> None:
    client = MlflowClient(f"sqlite:///{ROOT / 'mlflow.db'}")
    rows = [load_run(path, client) for path in sorted(CONFIG_ROOT.glob("*.yaml"))]
    if len(rows) != 5:
        raise RuntimeError(f"expected five completed ablation rows, found {len(rows)}")
    fold_fields = ["stage", "experiment_id", "feature_combination", "feature_families", "representation", "model", "hidden_dim", "fold", "seed", "status", "selected_epoch", "selected_val_uar", "checkpoint_path", "mlflow_run_id", "config_path", "runtime_directory", "source_commit", "evidence_ok", "val_overall_uar", "test_overall_uar"]
    fold_fields += [f"{split}_{component}_{metric}" for split in ("val", "test") for component in COMPONENTS for metric in METRICS]
    write_csv(ROOT / "reports/all_results_by_fold.csv", [{field: row[field] for field in fold_fields} for row in rows])
    summary = {}
    for split in ("val", "test"):
        values = [float(row[f"{split}_overall_uar"]) for row in rows]
        summary[f"{split}_mean"] = statistics.mean(values)
        summary[f"{split}_sd"] = statistics.stdev(values)
    summary_row = {"stage": "ablation", "experiment_id": "AC_mean_static_gated64", "completed_folds": 5, "evidence_status": "COMPLETE", "val_overall_uar_mean": summary["val_mean"], "val_overall_uar_sample_sd": summary["val_sd"], "test_overall_uar_mean": summary["test_mean"], "test_overall_uar_sample_sd": summary["test_sd"]}
    for split in ("val", "test"):
        for component in COMPONENTS:
            for metric in METRICS:
                values = [float(row[f"{split}_{component}_{metric}"]) for row in rows]
                summary_row[f"{split}_{component}_{metric}_mean"] = statistics.mean(values)
                summary_row[f"{split}_{component}_{metric}_sample_sd"] = statistics.stdev(values)
    write_csv(ROOT / "reports/all_results_summary.csv", [summary_row])
    write_csv(ROOT / "reports/run_ledger.csv", rows)
    old = existing_summary()
    comparisons = [
        ("AC Static Gated32", old["gated_AC_mean_gated32"]),
        ("ABCD Static Gated32", old["gated_ABCD_mean_gated32"]),
        ("ABCD Static Gated64 Equal", old["mtl_ABCD_mean_64_equal"]),
    ]
    lines = ["# CAREWSM AC Static Gated64 Ablation", "", "## Scope", "", "Exactly one isolated five-fold configuration was executed: `AC_mean_static_gated64`. No historical run or completed `audio_turns_combined` slot was rerun.", "", "Critical correction: `pooling: mean` is applied by the Dataset before the historical E3 model. E3 therefore receives a singleton `[B, 1, 2048]` representation; it is not meaningful turn-level temporal modeling. E3 differs from the newer static model mainly through architecture, normalization, gate inputs, dropout placement, and related implementation details.", "", "## Ablation result", "", f"VAL UAR: **{summary['val_mean']:.6f} ± {summary['val_sd']:.6f}**", f"Descriptive TEST UAR: **{summary['test_mean']:.6f} ± {summary['test_sd']:.6f}**", "", "| Fold | VAL UAR | TEST UAR | Epoch | Checkpoint | MLflow run ID |", "|---:|---:|---:|---:|---|---|"]
    for row in rows:
        lines.append(f"| {row['fold']} | {float(row['val_overall_uar']):.4f} | {float(row['test_overall_uar']):.4f} | {row['selected_epoch']} | `{row['checkpoint_path']}` | `{row['mlflow_run_id']}` |")
    lines += ["", "## Corpus/task metrics", "", "| Corpus/task | VAL UAR | TEST UAR | VAL Patient/Healthy | TEST Patient/Healthy |", "|---|---:|---:|---:|---:|"]
    labels = {"care_depression": "CARE Depression", "care_parkinson": "CARE Parkinson", "wsm_depression": "WSM Depression", "wsm_parkinson": "WSM Parkinson"}
    for component, label in labels.items():
        lines.append(f"| {label} | {summary_row[f'val_{component}_macro_recall_mean']:.4f} | {summary_row[f'test_{component}_macro_recall_mean']:.4f} | {summary_row[f'val_{component}_patient_recall_mean']:.4f} / {summary_row[f'val_{component}_healthy_recall_mean']:.4f} | {summary_row[f'test_{component}_patient_recall_mean']:.4f} / {summary_row[f'test_{component}_healthy_recall_mean']:.4f} |")
    lines += ["", "## Matched-fold comparisons", "", "| Comparison | VAL delta | TEST delta | Paired VAL differences | Component-safety result |", "|---|---:|---:|---|---|"]
    for label, baseline in comparisons:
        base_rows = list(csv.DictReader(Path("logs/audio_turns_combined/reports/all_results_by_fold.csv").open(encoding="utf-8")))
        exp = {"AC Static Gated32": "gated_AC_mean_gated32", "ABCD Static Gated32": "gated_ABCD_mean_gated32", "ABCD Static Gated64 Equal": "mtl_ABCD_mean_64_equal"}[label]
        base = sorted([r for r in base_rows if r["experiment_id"] == exp], key=lambda r: int(r["fold"]))
        val_diffs = [float(row["val_overall_uar"]) - float(b["val_overall_uar"]) for row, b in zip(rows, base)]
        test_diffs = [float(row["test_overall_uar"]) - float(b["test_overall_uar"]) for row, b in zip(rows, base)]
        component_drops = []
        for component in COMPONENTS:
            component_drops.append(statistics.mean(float(b[f"val_{component}_macro_recall"]) - float(row[f"val_{component}_macro_recall"]) for row, b in zip(rows, base)))
        safety = max(component_drops) <= 0.020
        lines.append(f"| {label} | {statistics.mean(val_diffs):+.4f} | {statistics.mean(test_diffs):+.4f} | {', '.join(f'{x:+.4f}' for x in val_diffs)} | {'PASS' if safety else 'FAIL'}; max mean component drop {max(component_drops):+.4f} |")
    e3 = [0.8589, 0.8343, 0.7972, 0.8026, 0.8724]
    e3_diffs = [float(row["val_overall_uar"]) - value for row, value in zip(rows, e3)]
    lines += ["", "## Historical E3 comparison", "", f"E3 VAL UAR was `0.8331 ± 0.0333`; AC Static Gated64 is `{summary['val_mean']:.4f} ± {summary['val_sd']:.4f}`. The matched fold differences (AC/64 minus E3) are {', '.join(f'{x:+.4f}' for x in e3_diffs)}, mean `{statistics.mean(e3_diffs):+.4f}`. This is an architectural comparison, not a pure hidden-size comparison.", "", "E3 remains **RETAINED — HIGH-VAL MTL CANDIDATE**. Its historical acceptance status remains REJECTED because its original improvement gate was not met.", "", "## Consolidated research leaderboard", "", "| Candidate | Category | VAL UAR | Descriptive TEST UAR | Status |", "|---|---|---:|---:|---|", "| TURNLESS-MEAN E3 | Historical joint multitask gated fusion | 0.8331 ± 0.0333 | 0.7573 ± 0.0211 | RETAINED — HIGH-VAL MTL CANDIDATE; historical gate REJECTED |", "| Native-turn DB-MTL | Historical joint multitask | 0.8317 ± 0.0371 | 0.784105 ± 0.022360 | Historical reference |", "| ABCD Static Gated64 Equal | Current joint multitask static gated fusion | 0.830855 ± 0.040364 | 0.782235 ± 0.022355 | VAL-selected current control |", "| Weighted-F native-turn | Historical joint multitask control | 0.8271 ± 0.0279 | 0.7643 ± 0.0231 | Historical control |", "| Independent Single-Task H1 | Separately trained Depression/Parkinson models paired by fold | 0.834070 ± 0.032738 | 0.782660 ± 0.025709 | Separate category; historical gate REJECTED |", "", "H1 is not a single jointly trained multitask model and is therefore not ranked as architecture-equivalent to E3, DB-MTL, or Static Gated64 Equal.", "", "## Decision", "", "A claimed substantive improvement requires at least +0.010 mean VAL UAR and no component mean VAL deterioration greater than 0.020. The AC/64 result is evaluated against AC/32 for the hidden-dimension question and against ABCD/64 for the feature-set comparison. TEST is descriptive only; no statistical significance is claimed from five folds.", "", "## Artifacts", "", "- `logs/audio_turns_combined_ablation/reports/all_results_by_fold.csv`", "- `logs/audio_turns_combined_ablation/reports/all_results_summary.csv`", "- `logs/audio_turns_combined_ablation/reports/run_ledger.csv`", "- `logs/audio_turns_combined_ablation/mlflow.db`", "- `configs/audio_turns_combined_ablation/ablation/`", "- `scripts/audio_turns_combined_ablation/run_all.sh`", "- `scripts/audio_turns_combined_ablation/run_worker.py`"]
    report = "\n".join(lines) + "\n"
    (Path("docs/evidence/audio_turns_combined_ablation") / "FINAL_RESEARCH_REPORT_EN.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
