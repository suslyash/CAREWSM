#!/usr/bin/env python3
"""Export historical validation-selected TEST metrics without running models."""

from __future__ import annotations

import csv
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from mlflow.tracking import MlflowClient

TASKS = ("depression", "parkinson")
BASE_METRICS = ("macro_recall", "patient_recall", "healthy_recall")
CORPORA = ("care", "wsm")
EXPERIMENTS = {
    "DEEP-003": "native_turns",
    "DEEP-006": "native_turn_fusion",
}


@dataclass(frozen=True)
class Entry:
    block: str
    experiment: str
    configuration: str
    model: str
    fold: int
    config: Path
    run_dir: Path
    checkpoint: Path
    epoch: int
    val_monitor: float
    values: dict[str, float]
    mlflow_db: str
    mlflow_run_id: str


def finite(value: str, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}")
    return result


def summary_row(summary: Path) -> tuple[dict[str, str], int, float]:
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    if not rows:
        raise ValueError(f"empty summary: {summary}")
    required = {"epoch", "dev_mean_macro_recall"}
    if not required <= set(rows[0]):
        raise ValueError(f"{summary}: missing validation monitor")
    selected = max(
        rows,
        key=lambda row: (
            finite(row["dev_mean_macro_recall"], "dev_mean_macro_recall"),
            -int(row["epoch"]),
        ),
    )
    epoch = int(selected["epoch"])
    monitor = finite(selected["dev_mean_macro_recall"], "dev_mean_macro_recall")
    return selected, epoch, monitor


def checkpoint_for(run_dir: Path, epoch: int) -> Path:
    candidates = sorted(
        path
        for path in (run_dir / "checkpoints").glob("*.pt")
        if path.name != "last.pt" and re.search(rf"epoch={epoch}(?:_|\.)", path.name)
    )
    if len(candidates) != 1:
        raise ValueError(
            f"{run_dir}: expected one selected checkpoint for epoch {epoch}"
        )
    return candidates[0]


def metric_values(row: dict[str, str]) -> dict[str, float]:
    values = {}
    for corpus in CORPORA:
        for task in TASKS:
            for metric in BASE_METRICS:
                key = f"test_{corpus}_{task}_{metric}"
                values[f"{corpus}_{task}_{metric}"] = finite(row[key], key)
    return values


def mlflow_id(db: str, run_name: str) -> str:
    client = MlflowClient(tracking_uri=db)
    experiments = [
        experiment.experiment_id for experiment in client.search_experiments()
    ]
    matches = []
    for experiment_id in experiments:
        matches.extend(
            run
            for run in client.search_runs([experiment_id])
            if run.data.tags.get("mlflow.runName") == run_name
        )
    if len(matches) != 1:
        raise ValueError(f"{run_name}: expected one MLflow run, found {len(matches)}")
    if matches[0].info.status != "FINISHED":
        raise ValueError(f"{run_name}: MLflow status is {matches[0].info.status}")
    return matches[0].info.run_id


def add_entry(
    root: Path,
    block: str,
    experiment: str,
    configuration: str,
    model: str,
    config: Path,
    run_dir: Path,
    db: str,
) -> Entry:
    summary = run_dir / "summary.txt"
    row, epoch, monitor = summary_row(summary)
    checkpoint = checkpoint_for(run_dir, epoch)
    if "Training complete" not in (run_dir / "train.log").read_text(encoding="utf-8"):
        raise ValueError(f"{run_dir}: missing Training complete")
    match = re.search(r"fold(\d+)_", run_dir.name)
    if match is None:
        raise ValueError(f"{run_dir}: cannot determine fold")
    fold = int(match.group(1))
    return Entry(
        block=block,
        experiment=experiment,
        configuration=configuration,
        model=model,
        fold=fold,
        config=config.relative_to(root),
        run_dir=run_dir.relative_to(root),
        checkpoint=checkpoint.relative_to(root),
        epoch=epoch,
        val_monitor=monitor,
        values=metric_values(row),
        mlflow_db=db,
        mlflow_run_id=mlflow_id(db, run_dir.name),
    )


def historical_entries(root: Path) -> list[Entry]:
    entries = []
    for experiment, kind in EXPERIMENTS.items():
        for parent in sorted(
            (root / "logs/audio_deep_features").glob(f"audio_deep_features_{kind}_*")
        ):
            if not parent.is_dir():
                continue
            configuration = parent.name.removeprefix(f"audio_deep_features_{kind}_")
            for model in ("mlp", "mdformer"):
                for fold in range(5):
                    config_name = f"{model}_fold{fold}.yaml"
                    config = (
                        root
                        / "configs/audio_deep_features"
                        / kind
                        / configuration
                        / config_name
                    )
                    runs = sorted(
                        parent.glob(
                            f"*_{model if kind == 'native_turns' else {'mlp': 'pooled_mlp', 'mdformer': 'md_former'}[model]}_fold{fold}_*/summary.txt"
                        )
                    )
                    if len(runs) != 1:
                        raise ValueError(
                            f"{parent}: expected one {model} fold {fold}, found {len(runs)}"
                        )
                    entries.append(
                        add_entry(
                            root,
                            f"{experiment}_{configuration}_{model}",
                            experiment,
                            configuration,
                            model,
                            config,
                            runs[0].parent,
                            "sqlite:///logs/audio_deep_features/mlflow.db",
                        )
                    )
    return entries


def deep008_entries(root: Path) -> list[Entry]:
    entries = []
    pilot = root / "logs/audio_representation_pilot/audio_representation_pilot_deep008"
    for block, config_stem, model in (
        ("DEEP-008_A", "A_native_mlp64", "mlp"),
        ("DEEP-008_F", "F_gated_emotion_trillsson_native64", "gated_fusion"),
    ):
        for fold in range(5):
            config = (
                root
                / "configs/audio_representation_pilot/deep008"
                / f"{config_stem}.yaml"
            )
            if fold:
                config = (
                    root
                    / "configs/audio_representation_pilot/deep008/followup_af"
                    / f"{config_stem}_fold{fold}.yaml"
                )
            runs = sorted(
                p.parent for p in pilot.glob(f"{config_stem}_fold{fold}_*/summary.txt")
            )
            if len(runs) != 1:
                raise ValueError(
                    f"{block} fold {fold}: expected one run, found {len(runs)}"
                )
            entries.append(
                add_entry(
                    root,
                    block,
                    "DEEP-008",
                    block,
                    model,
                    config,
                    runs[0],
                    "sqlite:///logs/audio_representation_pilot/mlflow.db",
                )
            )
    weighted = (
        root
        / "logs/audio_representation_pilot/deep008_weighted_f/audio_representation_pilot_deep008_weighted_f"
    )
    for fold in range(5):
        config = (
            root
            / "configs/audio_representation_pilot/deep008/weighted_f"
            / f"F_weighted_parkinson_fold{fold}.yaml"
        )
        runs = sorted(
            p.parent
            for p in weighted.glob(f"F_weighted_parkinson_fold{fold}_*/summary.txt")
        )
        if len(runs) != 1:
            raise ValueError(
                f"DEEP-008 weighted-F fold {fold}: expected one run, found {len(runs)}"
            )
        entries.append(
            add_entry(
                root,
                "DEEP-008_weighted-F",
                "DEEP-008",
                "weighted-F",
                "gated_fusion",
                config,
                runs[0],
                "sqlite:///logs/audio_representation_pilot/deep008_weighted_f/mlflow.db",
            )
        )
    stageg = (
        root
        / "logs/audio_representation_pilot/deep008_stageg_wsm_corrected/audio_representation_pilot_deep008_stageg_wsm_corrected"
    )
    equal_root = (
        root
        / "logs/audio_representation_pilot/deep008_weighted_f/audio_representation_pilot_deep008_weighted_f"
    )
    original_stageg = (
        root
        / "logs/audio_representation_pilot/deep008_stageg_wsm/audio_representation_pilot_deep008_stageg_wsm"
    )
    for method in (
        "equal",
        "stch",
        "progress_corrected",
        "ra_stch_corrected",
        "gradnorm_corrected",
        "pcgrad",
        "cagrad",
        "dbmtl",
    ):
        for fold in range(5):
            config_name = f"{method.removesuffix('_corrected')}_fold{fold}.yaml"
            config = (
                root
                / "configs/audio_representation_pilot/deep008/stageg_wsm"
                / config_name
            )
            run_root = stageg
            db = "sqlite:///logs/audio_representation_pilot/deep008_stageg_wsm_corrected/mlflow.db"
            if method in ("stch", "pcgrad", "cagrad", "dbmtl"):
                run_root = original_stageg
                db = "sqlite:///logs/audio_representation_pilot/deep008_stageg_wsm/mlflow.db"
            if method == "equal":
                config = (
                    root
                    / "configs/audio_representation_pilot/deep008/weighted_f"
                    / f"F_weighted_parkinson_fold{fold}.yaml"
                )
                run_root = equal_root
                db = "sqlite:///logs/audio_representation_pilot/deep008_weighted_f/mlflow.db"
                runs = sorted(
                    p.parent
                    for p in run_root.glob(
                        f"F_weighted_parkinson_fold{fold}_*/summary.txt"
                    )
                )
            else:
                runs = sorted(
                    p.parent
                    for p in run_root.glob(f"{method}_fold{fold}_*/summary.txt")
                )
            if not runs:
                raise ValueError(f"Stage G {method} fold {fold}: no run")
            run = runs[-1]
            entries.append(
                add_entry(
                    root,
                    f"DEEP-008_StageG_{method}",
                    "DEEP-008",
                    method,
                    "gated_fusion",
                    config,
                    run,
                    db,
                )
            )
    return entries


def write_per_fold(entries: list[Entry], output: Path) -> None:
    fields = [
        "report_block",
        "experiment",
        "configuration",
        "model",
        "fold",
        "selected_epoch",
        "val_monitor",
        "config",
        "run_dir",
        "checkpoint",
    ]
    fields += [
        f"{corpus}_{task}_{metric}"
        for corpus in CORPORA
        for task in TASKS
        for metric in BASE_METRICS
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            row = {
                "report_block": entry.block,
                "experiment": entry.experiment,
                "configuration": entry.configuration,
                "model": entry.model,
                "fold": entry.fold,
                "selected_epoch": entry.epoch,
                "val_monitor": f"{entry.val_monitor:.10f}",
                "config": str(entry.config),
                "run_dir": str(entry.run_dir),
                "checkpoint": str(entry.checkpoint),
            }
            row.update({key: f"{value:.10f}" for key, value in entry.values.items()})
            writer.writerow(row)


def write_manifest(entries: list[Entry], output: Path) -> None:
    fields = [
        "report_block",
        "experiment",
        "configuration",
        "model",
        "fold",
        "config",
        "checkpoint",
        "selected_epoch",
        "selection_monitor",
        "selection_value",
        "run_dir",
        "mlflow_db",
        "mlflow_run_id",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "report_block": entry.block,
                    "experiment": entry.experiment,
                    "configuration": entry.configuration,
                    "model": entry.model,
                    "fold": entry.fold,
                    "config": entry.config,
                    "checkpoint": entry.checkpoint,
                    "selected_epoch": entry.epoch,
                    "selection_monitor": "dev_mean_macro_recall",
                    "selection_value": f"{entry.val_monitor:.10f}",
                    "run_dir": entry.run_dir,
                    "mlflow_db": entry.mlflow_db,
                    "mlflow_run_id": entry.mlflow_run_id,
                }
            )


def write_aggregate(entries: list[Entry], output: Path) -> None:
    groups = {}
    for entry in entries:
        groups.setdefault(entry.block, []).append(entry)
    fields = ["report_block", "experiment", "configuration", "model", "task"]
    for corpus in ("care", "wsm", "average"):
        for metric in BASE_METRICS:
            fields += [f"{corpus}_{metric}_mean", f"{corpus}_{metric}_sd"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for block, rows in groups.items():
            if sorted(row.fold for row in rows) != list(range(5)):
                raise ValueError(f"{block}: incomplete folds")
            for task in TASKS + ("average",):
                row = {
                    "report_block": block,
                    "experiment": rows[0].experiment,
                    "configuration": rows[0].configuration,
                    "model": rows[0].model,
                    "task": (
                        "Depression"
                        if task == "depression"
                        else "PD" if task == "parkinson" else "Average"
                    ),
                }
                for corpus in ("care", "wsm", "average"):
                    for metric in BASE_METRICS:
                        values = []
                        for entry in rows:
                            if task == "average":
                                if corpus == "average":
                                    components = [
                                        entry.values[f"{source}_{task_name}_{metric}"]
                                        for source in CORPORA
                                        for task_name in TASKS
                                    ]
                                else:
                                    components = [
                                        entry.values[f"{corpus}_{task_name}_{metric}"]
                                        for task_name in TASKS
                                    ]
                                values.append(statistics.mean(components))
                            elif corpus == "average":
                                values.append(
                                    statistics.mean(
                                        [
                                            entry.values[f"{source}_{task}_{metric}"]
                                            for source in CORPORA
                                        ]
                                    )
                                )
                            else:
                                values.append(entry.values[f"{corpus}_{task}_{metric}"])
                        row[f"{corpus}_{metric}_mean"] = (
                            f"{statistics.mean(values):.10f}"
                        )
                        row[f"{corpus}_{metric}_sd"] = (
                            f"{statistics.stdev(values):.10f}"
                        )
                writer.writerow(row)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "docs/evidence/tables"
    entries = historical_entries(root) + deep008_entries(root)
    if len(entries) != 205 or len({entry.block for entry in entries}) != 41:
        raise ValueError(
            f"expected 205 entries/41 blocks, got {len(entries)}/{len({entry.block for entry in entries})}"
        )
    output.mkdir(parents=True, exist_ok=True)
    write_per_fold(entries, output / "DEEP008_HISTORICAL_TEST_METRICS_PER_FOLD.csv")
    write_aggregate(entries, output / "DEEP008_HISTORICAL_TEST_METRICS_AGGREGATED.csv")
    write_manifest(entries, output / "DEEP008_HISTORICAL_TEST_SOURCE_MANIFEST.csv")
    print("report_blocks=41 fold_entries=205 aggregate_rows=123")


if __name__ == "__main__":
    main()
