#!/usr/bin/env python3
"""Collect verified AV-005 fusion sweep results without touching runtimes."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sqlite3
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml


METRICS = (
    "macro_f1",
    "weighted_f1",
    "macro_recall",
    "weighted_recall",
    "patient_recall",
    "healthy_recall",
)
COMPONENTS = (
    "care_depression",
    "care_parkinson",
    "wsm_depression",
    "wsm_parkinson",
)
SPLITS = ("val", "test")
MONITOR = "val_mean_corpora_tasks_macro_recall"
CHECKPOINT_RE = re.compile(
    r"^epoch=(?P<epoch>\d+)_val_mean_corpora_tasks_macro_recall="
    r"(?P<value>[-+]?\d+(?:\.\d+)?)\.pt$"
)
EXPECTED_CONFIGS = 285
EXPECTED_VARIANTS = 57
VERIFIED_STATUSES = {"COMPLETED", "MISSING_METRICS"}


def finite(value: str | None) -> float | None:
    if value in (None, "", "nan", "NaN", "NA"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    atomic_write(path, buffer.getvalue())


def load_launcher_status(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return {row["run"]: row for row in csv.DictReader(handle, delimiter="\t")}


def parse_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = data["data"]["params"]
    run_name = data["experiment_info"]["params"]["run_name"]
    parts = run_name.split("__")
    model_token, feature_token, _ = parts
    feature_ids = {"audio": "", "video": "", "text": ""}
    for token in feature_token.split("_"):
        if token.startswith("A"):
            feature_ids["audio"] = "audio_id" + token[1:]
        elif token.startswith("V"):
            feature_ids["video"] = "video_" + token[1:]
        elif token.startswith("T"):
            feature_ids["text"] = "text_id" + token[1:]
    modalities = tuple(params["modalities"])
    variant_id = run_name.rsplit("__fold", 1)[0]
    return {
        "experiment_id": variant_id,
        "run_name": run_name,
        "model": data["model"]["name"],
        "architecture": model_token,
        "modalities": "+".join(modalities),
        "feature_ids": "+".join(feature_ids[m] for m in modalities),
        "audio_feature": feature_ids["audio"],
        "video_feature": feature_ids["video"],
        "text_feature": feature_ids["text"],
        "fold": str(params["fold"]),
        "seed": str(data.get("seed", "")),
        "config_path": str(path),
    }


def runtime_candidates(root: Path, run_name: str) -> list[Path]:
    runtime_root = root / "logs" / "fusion" / "sweep" / "sweep"
    if not runtime_root.is_dir():
        return []
    return sorted(
        path
        for path in runtime_root.iterdir()
        if path.is_dir() and (path.name == run_name or path.name.startswith(run_name + "_"))
    )


def mlflow_ids(db_path: Path) -> dict[str, str]:
    if not db_path.is_file():
        return {}
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = connection.execute(
            "SELECT r.run_uuid, t.value FROM runs r "
            "JOIN tags t ON t.run_uuid = r.run_uuid "
            "WHERE t.key = 'mlflow.runName'"
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        try:
            connection.close()
        except UnboundLocalError:
            pass
    return {name: run_id for run_id, name in rows}


def checkpoint(runtime: Path) -> tuple[Path | None, int | None, float | None, str | None]:
    files = sorted(
        path
        for path in (runtime / "checkpoints").glob("*.pt")
        if path.name != "last.pt"
    )
    if len(files) != 1:
        return None, None, None, f"expected exactly one selected checkpoint, found {len(files)}"
    if files[0].stat().st_size == 0:
        return None, None, None, "selected checkpoint is zero-byte"
    match = CHECKPOINT_RE.fullmatch(files[0].name)
    if match is None:
        return None, None, None, "selected checkpoint has unsupported identity"
    return (
        files[0],
        int(match.group("epoch")),
        float(match.group("value")),
        None,
    )


def metric_values(row: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    values: dict[str, Any] = {}
    missing = []
    for split in SPLITS:
        for component in COMPONENTS:
            for metric in METRICS:
                field = f"{split}_{component}_{metric}"
                value = finite(row.get(field))
                values[field] = value if value is not None else ""
                if value is None:
                    missing.append(field)
    return values, missing


def overall(values: dict[str, Any], split: str) -> float | None:
    recalls = [values.get(f"{split}_{component}_macro_recall") for component in COMPONENTS]
    return statistics.mean(recalls) if all(isinstance(value, float) for value in recalls) else None


def verify(config: dict[str, Any], root: Path, launcher: dict[str, dict[str, str]], ids: dict[str, str]) -> dict[str, Any]:
    result = dict(config)
    result["launcher_exit_code"] = launcher.get(config["run_name"], {}).get("exit_code", "")
    result["mlflow_run_id"] = ids.get(config["run_name"], "")
    result["source_commit"] = ""
    candidates = runtime_candidates(root, config["run_name"])
    result["runtime_directory"] = str(candidates[0]) if len(candidates) == 1 else ""
    if len(candidates) > 1:
        result.update(status="AMBIGUOUS", error_details="multiple matching runtime directories")
        return result
    if not candidates:
        launch_status = launcher.get(config["run_name"], {}).get("status", "")
        result.update(
            status="RUNNING" if launch_status == "RUNNING" else "PENDING",
            error_details="runtime directory not found",
        )
        return result
    runtime = candidates[0]
    train_log = runtime / "train.log"
    summary = runtime / "summary.txt"
    if not train_log.is_file() or "Training complete" not in train_log.read_text(
        encoding="utf-8", errors="replace"
    ):
        result.update(status="RUNNING", error_details="missing successful completion marker")
        return result
    if not summary.is_file():
        result.update(status="INCOMPLETE", error_details="missing summary.txt")
        return result
    selected, epoch, monitor, error = checkpoint(runtime)
    if error:
        result.update(status="INCOMPLETE", error_details=error)
        return result
    with summary.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    selected_rows = [row for row in rows if row.get("epoch") == str(epoch)]
    if len(selected_rows) != 1:
        result.update(status="INCOMPLETE", error_details="selected epoch is not unique in summary")
        return result
    row = selected_rows[0]
    recorded_monitor = finite(row.get(MONITOR))
    if recorded_monitor is None or abs(recorded_monitor - monitor) > 5e-5:
        result.update(status="INCOMPLETE", error_details="checkpoint and summary monitor disagree")
        return result
    values, missing = metric_values(row)
    result.update(
        status="MISSING_METRICS" if missing else "COMPLETED",
        error_details="missing: " + ",".join(missing) if missing else "",
        checkpoint_path=str(selected),
        selected_epoch=epoch,
        best_val_monitor=recorded_monitor,
        **values,
    )
    for split in SPLITS:
        value = overall(values, split)
        result[f"{split}_overall_uar"] = value if value is not None else ""
    return result


def metric_fields(prefix: str = "") -> list[str]:
    return [
        f"{prefix}{split}_{component}_{metric}"
        for split in SPLITS
        for component in COMPONENTS
        for metric in METRICS
    ]


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = sorted({row["experiment_id"] for row in rows})
    output = []
    for variant in variants:
        group = [
            row
            for row in rows
            if row["experiment_id"] == variant
            and row["status"] in VERIFIED_STATUSES
        ]
        base = next(row for row in rows if row["experiment_id"] == variant)
        item = {key: base.get(key, "") for key in ("experiment_id", "architecture", "model", "modalities", "feature_ids", "audio_feature", "video_feature", "text_feature")}
        item["completed_folds"] = len(group)
        item["five_fold_complete"] = "yes" if len(group) == 5 else "no"
        for field in ["val_overall_uar", "test_overall_uar"] + metric_fields():
            values = [float(row[field]) for row in group if row.get(field) not in ("", None)]
            item[f"{field}_mean"] = statistics.mean(values) if len(values) == 5 else ""
            item[f"{field}_sd"] = statistics.stdev(values) if len(values) == 5 else ""
        output.append(item)
    return output


def compare_guided(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["experiment_id"], row["fold"]): row
        for row in rows
        if row["status"] in VERIFIED_STATUSES
    }
    variants = sorted({row["experiment_id"] for row in rows if row["architecture"] in {"text_guided", "audio_guided"}})
    pairs = []
    for variant in variants:
        prefix, features = variant.split("__", 1)
        if prefix != "text_guided":
            continue
        audio_variant = "audio_guided__" + features
        for fold in map(str, range(5)):
            left = lookup.get((variant, fold))
            right = lookup.get((audio_variant, fold))
            item = {"feature_variant": features, "fold": fold, "text_guided_status": left is not None, "audio_guided_status": right is not None}
            for split in SPLITS:
                split_metrics = [
                    f"{split}_{component}_{metric}"
                    for component in COMPONENTS
                    for metric in METRICS
                ]
                for metric in [f"{split}_overall_uar"] + split_metrics:
                    item[f"{metric}_audio_minus_text"] = (
                        float(right[metric]) - float(left[metric]) if left and right and left.get(metric) not in ("", None) and right.get(metric) not in ("", None) else ""
                    )
            pairs.append(item)
    return pairs


def report(rows: list[dict[str, Any]], summaries: list[dict[str, Any]], source_commit: str) -> str:
    counts = {status: sum(row["status"] == status for row in rows) for status in ("COMPLETED", "MISSING_METRICS", "RUNNING", "PENDING", "FAILED", "INCOMPLETE", "AMBIGUOUS")}
    complete_variants = sum(row["five_fold_complete"] == "yes" for row in summaries)
    lines = [
        "# AV-005 Fusion Sweep Research Report", "",
        "`UPSTREAM_PROVENANCE=UNKNOWN — EXPLORATORY RESULTS ONLY`", "",
        "## A. Experiment completion", "",
        f"Expected jobs: **{len(rows)}** (matrix contract: {EXPECTED_CONFIGS}).", "",
        f"Verified completed folds: **{sum(counts.get(status, 0) for status in VERIFIED_STATUSES)}** (fully metric-complete: {counts['COMPLETED']}; missing metrics: {counts['MISSING_METRICS']}). Complete five-fold variants: **{complete_variants}/{EXPECTED_VARIANTS}**.", "",
        "Status counts: " + ", ".join(f"{key}={value}" for key, value in counts.items()) + ".", "",
        f"Collector source commit: `{source_commit}`. Runtime artifacts were read-only inputs.", "",
        "## B. Data and feature inventory", "",
        "Audio: `audio_id1`, `audio_id2`, `audio_id4`; video: `video_4`; text: `text_id9`, `text_id10`. The matrix contains 11 bimodal and 6 trimodal feature selections over five canonical folds.", "",
        "## C. VAL results", "",
        "Only complete five-fold variants are ranked. Values are mean ± sample SD of fold-level Overall UAR; TEST is never used for ranking.", "",
        "| Rank | Model | Features | VAL Overall UAR | TEST Overall UAR (descriptive) |", "|---:|---|---|---:|---:|",
    ]
    ranked = sorted((row for row in summaries if row["five_fold_complete"] == "yes"), key=lambda row: (-float(row["val_overall_uar_mean"]), row["experiment_id"]))
    for rank, row in enumerate(ranked, 1):
        val = f"{float(row['val_overall_uar_mean']):.4f} ± {float(row['val_overall_uar_sd']):.4f}"
        test = "missing"
        if row["test_overall_uar_mean"] != "":
            test = f"{float(row['test_overall_uar_mean']):.4f} ± {float(row['test_overall_uar_sd']):.4f}"
        lines.append(f"| {rank} | `{row['model']}` | `{row['feature_ids']}` | {val} | {test} |")
    lines += ["", "## D. TEST results — DESCRIPTIVE ONLY", "",
        "TEST values are reported only as statistics for the selected validation checkpoints and do not determine any decision.",
        "If all TEST metrics are missing, this report remains VAL-only until a separate selected-checkpoint TEST inference/evaluation step has produced verified metrics; training is already complete and must not be repeated.", "", "## E–F. Bimodal and trimodal comparisons", "", "The CSV outputs preserve all architecture, modality, feature-version, fold, task, corpus, split, and metric dimensions. Matched Text-Guided versus Audio-Guided fold differences are in `comparison_audio_vs_text_guided.csv`.", "", "## G. Best models", "", (f"Best validation variant: `{ranked[0]['experiment_id']}` with Overall UAR {float(ranked[0]['val_overall_uar_mean']):.4f} ± {float(ranked[0]['val_overall_uar_sd']):.4f}." if ranked else "No five-fold-complete validation variant is currently eligible."), "", "## H. Reproducibility and limitations", "", "The collector records config identity, runtime evidence, selected checkpoint, epoch, validation monitor, launcher exit code where available, MLflow run ID where available, and source provenance where available. Missing or incomplete artifacts are not fabricated. Results remain exploratory because upstream FIT/SELECTION provenance is unknown; no leakage-free or clinical-validity claim is made."]
    return "\n".join(lines) + "\n"


def collect(run_root: Path, matrix: str) -> dict[str, Any]:
    if matrix != "sweep":
        raise ValueError("only --matrix sweep is supported")
    config_root = run_root / "configs" / "fusion" / matrix
    configs = sorted(config_root.glob("*.yaml"))
    if len(configs) != EXPECTED_CONFIGS:
        raise ValueError(f"expected {EXPECTED_CONFIGS} configs, found {len(configs)}")
    parsed = [parse_config(path) for path in configs]
    launcher_root = run_root / "logs" / "fusion" / "sweep" / "launcher"
    launcher = load_launcher_status(launcher_root / "status.tsv")
    ids = mlflow_ids(run_root / "logs" / "fusion" / "sweep" / "mlflow.db")
    rows = [verify(config, run_root, launcher, ids) for config in parsed]
    rows.sort(key=lambda row: (row["experiment_id"], int(row["fold"])))
    summaries = aggregate(rows)
    reports = run_root / "logs" / "fusion" / "sweep" / "reports"
    status_fields = list(parsed[0]) + ["runtime_directory", "status", "launcher_exit_code", "checkpoint_path", "selected_epoch", "best_val_monitor", "mlflow_run_id", "source_commit", "error_details"] + metric_fields() + ["val_overall_uar", "test_overall_uar"]
    write_csv(reports / "runs_status.csv", rows, status_fields)
    fold_rows = [row for row in rows if row["status"] in VERIFIED_STATUSES]
    fold_fields = list(parsed[0]) + ["runtime_directory", "status", "launcher_exit_code", "checkpoint_path", "selected_epoch", "best_val_monitor", "mlflow_run_id", "source_commit"] + metric_fields() + ["val_overall_uar", "test_overall_uar"]
    write_csv(reports / "results_by_fold.csv", fold_rows, fold_fields)
    summary_fields = list(summaries[0]) if summaries else []
    write_csv(reports / "results_summary.csv", summaries, summary_fields)
    ranked = [row for row in summaries if row["five_fold_complete"] == "yes"]
    ranked.sort(key=lambda row: (-float(row["val_overall_uar_mean"]), row["experiment_id"]))
    ranking = []
    for rank, row in enumerate(ranked, 1):
        ranking.append({"rank": rank, **{key: row.get(key, "") for key in ("experiment_id", "model", "modalities", "audio_feature", "video_feature", "text_feature", "val_overall_uar_mean", "val_overall_uar_sd", "test_overall_uar_mean", "test_overall_uar_sd")}, "selection_basis": "mean five-fold VAL Overall UAR", "provenance": "UPSTREAM_PROVENANCE=UNKNOWN"})
    write_csv(reports / "ranking_val.csv", ranking, list(ranking[0]) if ranking else ["rank", "experiment_id"])
    comparisons = compare_guided(rows)
    write_csv(reports / "comparison_audio_vs_text_guided.csv", comparisons, list(comparisons[0]) if comparisons else ["feature_variant", "fold"])
    source_commit = "unknown"
    try:
        source_commit = subprocess.check_output(["git", "-C", str(run_root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    atomic_write(reports / "FINAL_RESEARCH_REPORT.md", report(rows, summaries, source_commit))
    return {"rows": rows, "summaries": summaries, "reports": reports}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, choices=("sweep",))
    parser.add_argument("--run-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = collect(args.run_root.resolve(), args.matrix)
    counts = {}
    for row in result["rows"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("collected " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    print(f"reports: {result['reports']}")


if __name__ == "__main__":
    main()
