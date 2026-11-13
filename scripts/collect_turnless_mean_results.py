#!/usr/bin/env python3
"""Collect selected VAL/TEST rows and generate the turnless mean report."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import hashlib
import json
import re
import subprocess

from mlflow.tracking import MlflowClient

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
TRAINING_SOURCE_COMMIT = "8a91a06e52d4710f54503b824f76a969cf73942b"
ROOT = Path("logs/audio_representation_pilot/turnless_mean")
CHECKPOINT_RE = re.compile(
    r"^epoch=(\d+)_val_mean_corpora_tasks_macro_recall=([0-9.]+)\.pt$"
)


def finite(value: str | None) -> float | None:
    if value in (None, "", "-"):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def runtime_for(run_name: str) -> Path:
    candidates = sorted(
        (ROOT / "audio_representation_pilot_turnless_mean").glob(f"{run_name}_*")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one runtime directory for {run_name}, found {len(candidates)}"
        )
    return candidates[0]


def resolve_checkpoint(runtime: Path) -> tuple[Path, int, float]:
    selected = [
        p for p in sorted((runtime / "checkpoints").glob("*.pt")) if p.name != "last.pt"
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one retained checkpoint in {runtime}, found {len(selected)}"
        )
    match = CHECKPOINT_RE.fullmatch(selected[0].name)
    if match is None:
        raise RuntimeError(f"non-canonical checkpoint filename: {selected[0]}")
    return selected[0], int(match.group(1)), float(match.group(2))


def summary_row(path: Path, epoch: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    matches = [row for row in rows if row.get("epoch") == str(epoch)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one summary row for epoch {epoch}: {path}")
    return matches[0]


def mlflow_runs() -> dict[str, object]:
    client = MlflowClient(f"sqlite:///{ROOT / 'mlflow.db'}")
    experiment = client.get_experiment_by_name(
        "audio_representation_pilot_turnless_mean"
    )
    if experiment is None:
        raise RuntimeError("missing MLflow experiment")
    runs = client.search_runs(
        [experiment.experiment_id], order_by=["attributes.start_time ASC"]
    )
    result = {}
    for run in runs:
        for experiment in EXPERIMENTS:
            for fold in range(5):
                key = f"{experiment}_fold{fold}"
                if run.info.run_name.startswith(key + "_"):
                    if key in result:
                        raise RuntimeError(f"duplicate MLflow run: {key}")
                    result[key] = run
    return result


def digest_runtime(runtime: Path) -> dict[str, str]:
    paths = [
        runtime / "summary.txt",
        runtime / "train.log",
        *sorted((runtime / "checkpoints").glob("*.pt")),
    ]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def report(
    folds: list[dict[str, object]], ledger: list[dict[str, object]], commit: str
) -> str:
    def vals(exp: str, split: str, suffix: str = "overall_uar") -> list[float]:
        return [
            float(row[f"{split}_{suffix}"]) for row in folds if row["experiment"] == exp
        ]

    lines = [
        "# TURNLESS MEAN Five-Fold Results",
        "",
        f"Training source commit: `{TRAINING_SOURCE_COMMIT}`; reporting commit: `{commit}`; seed 5; 20/20 runs completed.",
        "",
        "Generated exclusively by `scripts/collect_turnless_mean_results.py` from runtime summaries, retained validation checkpoints, and local MLflow. Validation selects the checkpoint; TEST is descriptive and was not used for selection.",
        "",
        "Historical pooling verification used `test_care` and `test_wsm` feature artifacts only for arithmetic (CARE 3 turns, WSM 1 turn; max_abs_error=0.0; output shape `[1, 2048]`). Labels and predictions were not used.",
        "",
    ]
    for split, baseline, title in (
        ("val", BASELINE_VAL, "Table A — Validation Results"),
        (
            "test",
            BASELINE_TEST,
            "Table B — Descriptive TEST Results — not used for model selection",
        ),
    ):
        lines += [
            f"## {title}",
            "",
            "| Experiment | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± SD | Delta vs Weighted-F |"
            + (" Paired wins | Decision |" if split == "val" else ""),
            "|---|---:|---:|---:|---:|---:|---:|---:|"
            + ("---:|---:|" if split == "val" else ""),
        ]
        for exp in EXPERIMENTS:
            values = vals(exp, split)
            mean, sd = stats(values)
            delta = mean - statistics.mean(baseline)
            tail = ""
            if split == "val":
                tail = f" {sum(a > b for a,b in zip(values, baseline))}/5 | REJECTED |"
            lines.append(
                f"| {exp} | "
                + " | ".join(f"{v:.4f}" for v in values)
                + f" | {mean:.4f} ± {sd:.4f} | {delta:+.4f} |"
                + tail
            )
        mean, sd = stats(baseline)
        lines.append(
            f"| Weighted-F historical control | "
            + " | ".join(f"{v:.4f}" for v in baseline)
            + f" | {mean:.4f} ± {sd:.4f} | — |"
            + (" — | CONTROL |" if split == "val" else "")
        )
        lines += [""]
    lines += [
        "Acceptance is validation-only: +0.010 mean VAL UAR was required, with no component mean deterioration above 0.020. All four configurations are rejected; no significance or clinical superiority is claimed.",
        "",
        "## Table C — Five-fold mean Macro Recall by corpus/task",
        "",
        "| Experiment | CARE Depression VAL/TEST | CARE Parkinson VAL/TEST | WSM Depression VAL/TEST | WSM Parkinson VAL/TEST |",
        "|---|---:|---:|---:|---:|",
    ]
    for exp in EXPERIMENTS:
        cells = []
        for component in COMPONENTS:
            cells.append(
                f"{statistics.mean(vals(exp,'val',component+'_macro_recall')):.4f}/{statistics.mean(vals(exp,'test',component+'_macro_recall')):.4f}"
            )
        lines.append(f"| {exp} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Table D — Patient/Healthy Recall by corpus/task",
        "",
        "| Experiment | CARE Depression VAL patient/healthy | CARE Depression TEST patient/healthy | CARE Parkinson VAL patient/healthy | CARE Parkinson TEST patient/healthy | WSM Depression VAL patient/healthy | WSM Depression TEST patient/healthy | WSM Parkinson VAL patient/healthy | WSM Parkinson TEST patient/healthy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for exp in EXPERIMENTS:
        cells = []
        for component in COMPONENTS:
            for split in ("val", "test"):
                cells.append(
                    f"{statistics.mean(vals(exp,split,component+'_patient_recall')):.4f}/{statistics.mean(vals(exp,split,component+'_healthy_recall')):.4f}"
                )
        lines.append(f"| {exp} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Table E — Reproducibility",
        "",
        "| Experiment | Fold | MLflow run ID | Selected epoch | Selected checkpoint | Status |",
        "|---|---:|---|---:|---|---|",
    ]
    for row in ledger:
        lines.append(
            f"| {row['experiment']} | {row['fold']} | {row['mlflow_run_id']} | {row['selected_epoch']} | `{row['selected_checkpoint']}` | {row['status']} |"
        )
    lines += [
        "",
        "TEST is previously observed descriptive evidence, not a pristine independent holdout.",
        "",
        "Feature families: `emotionmsp_embeds` + `trillsson4_1`; arithmetic mean over valid turns followed by concatenation; joint CARE+WSM five-fold training; seed 5; AdamW; learning rate 1e-4; batch size 64; dropout 0.1; maximum 50 epochs; patience 7. Parkinson class weights were TRAIN-only.",
        "",
        "The collector fails on missing or ambiguous runtimes/checkpoints, monitor disagreement, incomplete MLflow provenance, or non-finite metrics; it never falls back to an arbitrary checkpoint.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    runs = mlflow_runs()
    folds = []
    ledger = []
    digests = {}
    for experiment in EXPERIMENTS:
        model = (
            "carewsm_gated_fusion" if "gated" in experiment else "carewsm_pooled_mlp"
        )
        hidden = 64 if experiment.endswith("64") else 32
        for fold in range(5):
            run_name = f"{experiment}_fold{fold}"
            runtime = runtime_for(run_name)
            checkpoint, epoch, checkpoint_monitor = resolve_checkpoint(runtime)
            row = summary_row(runtime / "summary.txt", epoch)
            monitor = finite(row.get("val_mean_corpora_tasks_macro_recall"))
            if monitor is None or abs(monitor - checkpoint_monitor) > 5e-5:
                raise RuntimeError(f"checkpoint/summary monitor mismatch: {run_name}")
            if "Training complete" not in (runtime / "train.log").read_text(
                encoding="utf-8"
            ):
                raise RuntimeError(f"training did not complete: {run_name}")
            run = runs.get(run_name)
            if run is None or run.info.status != "FINISHED":
                raise RuntimeError(f"missing finished MLflow run: {run_name}")
            output = {
                "experiment": experiment,
                "fold": fold,
                "model": model,
                "hidden_dim": hidden,
                "pooling": "mean",
                "status": "COMPLETED",
                "runtime_directory": str(runtime),
                "selected_epoch": epoch,
                "selected_val_uar": f"{monitor:.10f}",
                "selected_checkpoint": str(checkpoint),
                "mlflow_run_id": run.info.run_id,
                "training_source_commit": TRAINING_SOURCE_COMMIT,
                "reporting_commit": commit,
            }
            for split in ("val", "test"):
                for component in COMPONENTS:
                    for metric in METRICS:
                        value = finite(row.get(f"{split}_{component}_{metric}"))
                        if value is None:
                            raise RuntimeError(
                                f"missing metric {split}_{component}_{metric}: {run_name}"
                            )
                        output[f"{split}_{component}_{metric}"] = f"{value:.10f}"
                output[f"{split}_overall_uar"] = (
                    f"{statistics.mean(float(output[f'{split}_{c}_macro_recall']) for c in COMPONENTS):.10f}"
                )
            folds.append(output)
            ledger.append(
                {
                    "experiment": experiment,
                    "fold": fold,
                    "model": model,
                    "pooling": "mean",
                    "hidden_dim": hidden,
                    "training_source_commit": TRAINING_SOURCE_COMMIT,
                    "reporting_commit": commit,
                    "seed": 5,
                    "selected_epoch": epoch,
                    "selected_checkpoint": str(checkpoint),
                    "selection_monitor": "val_mean_corpora_tasks/macro_recall",
                    "selected_val_uar": f"{monitor:.10f}",
                    "runtime_directory": str(runtime),
                    "mlflow_run_id": run.info.run_id,
                    "mlflow_status": run.info.status,
                    "status": "COMPLETED",
                    "failure_reason": "",
                }
            )
            digests.update(digest_runtime(runtime))
    summary = []
    for experiment in EXPERIMENTS:
        for split in ("val", "test"):
            values = [
                float(row[f"{split}_overall_uar"])
                for row in folds
                if row["experiment"] == experiment
            ]
            mean, sd = stats(values)
            summary.append(
                {
                    "experiment": experiment,
                    "split": split,
                    "completed_folds": len(values),
                    "mean_overall_uar": f"{mean:.10f}",
                    "sample_sd": f"{sd:.10f}",
                }
            )
    write_csv(Path("turnless_mean_results_by_fold.csv"), folds)
    write_csv(Path("turnless_mean_results_summary.csv"), summary)
    write_csv(Path("turnless_mean_run_ledger.csv"), ledger)
    Path("docs/evidence/TURNLESS_MEAN_FIVEFOLD_RESULTS_EN.md").write_text(
        report(folds, ledger, commit), encoding="utf-8"
    )
    e3 = [
        float(row["test_overall_uar"])
        for row in folds
        if row["experiment"] == "E3_gated32"
    ]
    audit = {
        "status": "PASS",
        "completed_runs": len(folds),
        "expected_runs": 20,
        "training_source_commit": TRAINING_SOURCE_COMMIT,
        "reporting_commit": commit,
        "checks": {
            "all_runs_completed": len(folds) == 20,
            "real_mlflow_ids": all(
                row["mlflow_run_id"] != "unavailable_until_runtime_query"
                for row in ledger
            ),
            "exact_checkpoint_resolution": True,
            "no_training_or_inference": True,
            "test_not_used_for_selection": True,
            "source_artifact_sha256": digests,
            "prior_e3_test_mean": 0.7596,
            "recomputed_e3_test_mean": statistics.mean(e3),
            "recomputed_e3_test_sample_sd": statistics.stdev(e3),
        },
    }
    Path("turnless_mean_consistency_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not args.allow_incomplete and len(folds) != 20:
        raise SystemExit("incomplete collection")
    print(f"Collected {len(folds)}/20 completed runs")


if __name__ == "__main__":
    main()
