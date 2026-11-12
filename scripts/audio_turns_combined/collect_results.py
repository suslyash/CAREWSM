#!/usr/bin/env python3
"""Collect verified runtime summaries into deterministic program reports."""

from __future__ import annotations

import csv
import math
import re
import statistics
import subprocess
from pathlib import Path

import yaml
from mlflow.tracking import MlflowClient

ROOT = Path("logs/audio_turns_combined")
CONFIG_ROOT = Path("configs/audio_turns_combined")
COMPONENTS = ("care_depression", "care_parkinson", "wsm_depression", "wsm_parkinson")
METRICS = (
    "macro_f1",
    "weighted_f1",
    "macro_recall",
    "weighted_recall",
    "patient_recall",
    "healthy_recall",
)
CHECKPOINT = re.compile(
    r"^epoch=(\d+)_val_mean_corpora_tasks_macro_recall=([0-9.]+)\.pt$"
)
FIELDS = (
    "stage",
    "experiment_id",
    "feature_combination",
    "feature_families",
    "representation",
    "model",
    "hidden_dim",
    "fold",
    "seed",
    "status",
    "attempt_number",
    "start_time",
    "end_time",
    "exit_code",
    "failure_reason",
    "runtime_directory",
    "checkpoint_path",
    "selected_epoch",
    "selected_val_uar",
    "mlflow_run_id",
    "source_commit",
)


def configs() -> list[Path]:
    return sorted(CONFIG_ROOT.glob("*/*.yaml"))


def runtime(run_name: str) -> Path | None:
    candidates = sorted((ROOT / "audio_turns_combined").glob(f"{run_name}_*"))
    return candidates[0] if len(candidates) == 1 else None


def parse_config(path: Path) -> dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = data["data"]["params"]
    info = data["experiment_info"]["params"]
    configured_families = params["feature_family"]
    families = [configured_families] if isinstance(configured_families, str) else configured_families
    model = data["model"]
    code = "".join(
        {
            "emotionmsp_embeds": "A",
            "emotionmsp_trajectories": "B",
            "trillsson4_1": "C",
            "wav2vec_xlsr300m": "D",
        }[family]
        for family in families
    )
    return {
        "stage": path.parent.name,
        "experiment_id": info["run_name"].rsplit("_fold", 1)[0],
        "feature_combination": code,
        "feature_families": "+".join(families),
        "representation": params["pooling"],
        "model": model["name"],
        "hidden_dim": model["params"]["hidden_dim"],
        "fold": params["fold"],
        "seed": data["seed"],
        "run_name": info["run_name"],
        "config_path": str(path),
    }


def mlflow_ids() -> dict[str, str]:
    client = MlflowClient(f"sqlite:///{ROOT / 'mlflow.db'}")
    experiment = client.get_experiment_by_name("audio_turns_combined")
    if experiment is None:
        return {}
    result = {}
    names = {
        yaml.safe_load(path.read_text(encoding="utf-8"))["experiment_info"]["params"][
            "run_name"
        ]
        for path in configs()
    }
    for run in client.search_runs([experiment.experiment_id]):
        for name in names:
            if run.info.run_name.startswith(name + "_"):
                result[name] = run.info.run_id
    return result


def verified(
    config: dict[str, object], ids: dict[str, str]
) -> dict[str, object] | None:
    run_dir = runtime(str(config["run_name"]))
    if (
        run_dir is None
        or not (run_dir / "summary.txt").is_file()
        or "Training complete"
        not in (run_dir / "train.log").read_text(encoding="utf-8")
    ):
        return None
    checkpoints = [
        path
        for path in (run_dir / "checkpoints").glob("*.pt")
        if path.name != "last.pt"
    ]
    if len(checkpoints) != 1:
        return None
    match = CHECKPOINT.fullmatch(checkpoints[0].name)
    if match is None or str(config["run_name"]) not in ids:
        return None
    with (run_dir / "summary.txt").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    selected = [row for row in rows if row.get("epoch") == match.group(1)]
    if (
        len(selected) != 1
        or abs(
            float(selected[0]["val_mean_corpora_tasks_macro_recall"])
            - float(match.group(2))
        )
        > 5e-5
    ):
        return None
    row = selected[0]
    result = dict(config)
    result.update(
        {
            "status": "COMPLETED",
            "runtime_directory": str(run_dir),
            "checkpoint_path": str(checkpoints[0]),
            "selected_epoch": int(match.group(1)),
            "selected_val_uar": float(row["val_mean_corpora_tasks_macro_recall"]),
            "mlflow_run_id": ids[str(config["run_name"])],
        }
    )
    for split in ("val", "test"):
        values = []
        for component in COMPONENTS:
            for metric in METRICS:
                result[f"{split}_{component}_{metric}"] = float(
                    row[f"{split}_{component}_{metric}"]
                )
            values.append(result[f"{split}_{component}_macro_recall"])
        result[f"{split}_overall_uar"] = statistics.mean(values)
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    ids = mlflow_ids()
    rows = []
    for path in configs():
        config = parse_config(path)
        item = verified(config, ids)
        if item is not None:
            item["source_commit"] = commit
            rows.append(item)
    rows.sort(
        key=lambda row: (str(row["stage"]), str(row["experiment_id"]), int(row["fold"]))
    )
    if not rows:
        return
    fold_fields = list(rows[0])
    write_csv(
        ROOT / "reports/all_results_by_fold.csv",
        [
            {
                field: row.get(field, "")
                for field in fold_fields
                if field != "run_name" and field != "config_path"
            }
            for row in rows
        ],
    )
    summary = []
    for experiment in sorted({str(row["experiment_id"]) for row in rows}):
        selected = [row for row in rows if row["experiment_id"] == experiment]
        for split in ("val", "test"):
            values = [row[f"{split}_overall_uar"] for row in selected]
            summary.append(
                {
                    "experiment_id": experiment,
                    "stage": selected[0]["stage"],
                    "completed_folds": len(values),
                    "mean_overall_uar": (
                        statistics.mean(values) if len(values) == 5 else ""
                    ),
                    "sample_sd": statistics.stdev(values) if len(values) == 5 else "",
                    "split": split,
                }
            )
    write_csv(ROOT / "reports/all_results_summary.csv", summary)
    lines = [
        "# audio_turns_combined Research Report",
        "",
        f"Source commit: `{commit}`",
        "",
        f"Verified completed runs: {len(rows)}",
        "",
        "## Table A — VAL",
        "",
        "| Stage | Experiment | Fold | Overall UAR | CARE Depression | CARE Parkinson | WSM Depression | WSM Parkinson |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['experiment_id']} | {row['fold']} | {row['val_overall_uar']:.4f} | "
            + " | ".join(
                f"{row[f'val_{component}_macro_recall']:.4f}"
                for component in COMPONENTS
            )
            + " |"
        )
    lines += [
        "",
        "## Table B — DESCRIPTIVE TEST — NOT USED FOR SELECTION",
        "",
        "| Stage | Experiment | Fold | Overall UAR | CARE Depression | CARE Parkinson | WSM Depression | WSM Parkinson |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['experiment_id']} | {row['fold']} | {row['test_overall_uar']:.4f} | "
            + " | ".join(
                f"{row[f'test_{component}_macro_recall']:.4f}"
                for component in COMPONENTS
            )
            + " |"
        )
    lines += [
        "",
        "## Table C — Class-Specific Recall",
        "",
        "All six metrics are present in `all_results_by_fold.csv`; Patient Recall and Healthy Recall are recorded separately for every corpus/task/split.",
        "",
        "## Table D — Stage Decisions",
        "",
        "Stage candidate selection is based only on complete five-fold mean validation Macro Recall. Incomplete configurations are excluded.",
        "",
        "## Table E — Reproducibility",
        "",
        "The run ledger records source commit, config path, runtime directory, MLflow ID, selected epoch, checkpoint, exit status, and failure fields. TEST is descriptive only.",
    ]
    (ROOT / "reports/all_results_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (
        Path("docs/evidence/audio_turns_combined") / "FINAL_RESEARCH_REPORT_EN.md"
    ).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
