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


def require_one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"{label}: expected exactly one run, found {len(paths)}")
    return paths[0]


def completed_run_candidates(paths: list[Path]) -> list[Path]:
    return [
        path
        for path in paths
        if (path / "train.log").is_file()
        and (path / "summary.txt").is_file()
        and "Training complete" in (path / "train.log").read_text(encoding="utf-8")
        and any(
            checkpoint.name != "last.pt"
            for checkpoint in (path / "checkpoints").glob("*.pt")
        )
    ]


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


def checkpoint_monitor(checkpoint: Path) -> tuple[int, float]:
    match = re.search(
        r"epoch=(?P<epoch>\d+)_(?:dev_mean_macro_recall|val_mean_corpora_tasks_macro_recall)=(?P<value>[-+]?\d+(?:\.\d+)?)\.pt$",
        checkpoint.name,
    )
    if match is None:
        raise ValueError(f"unsupported checkpoint identity: {checkpoint}")
    return int(match.group("epoch")), float(match.group("value"))


def metric_values(row: dict[str, str]) -> dict[str, float]:
    values = {}
    for corpus in CORPORA:
        for task in TASKS:
            for metric in BASE_METRICS:
                key = f"test_{corpus}_{task}_{metric}"
                value = finite(row[key], key)
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"out-of-range {key}: {value}")
                values[f"{corpus}_{task}_{metric}"] = value
            macro = values[f"{corpus}_{task}_macro_recall"]
            recall_mean = statistics.mean(
                [
                    values[f"{corpus}_{task}_patient_recall"],
                    values[f"{corpus}_{task}_healthy_recall"],
                ]
            )
            if abs(macro - recall_mean) > 5e-4:
                raise ValueError(f"inconsistent TEST recalls for {corpus}/{task}")
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
    checkpoint_epoch, checkpoint_value = checkpoint_monitor(checkpoint)
    if checkpoint_epoch != epoch or abs(checkpoint_value - monitor) > 5e-5:
        raise ValueError(f"{run_dir}: checkpoint does not identify selected VAL row")
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
                    run = require_one(
                        completed_run_candidates([item.parent for item in runs]),
                        f"{parent}: {model} fold {fold}",
                    )
                    entries.append(
                        add_entry(
                            root,
                            f"{experiment}_{configuration}_{model}",
                            experiment,
                            configuration,
                            model,
                            config,
                            run,
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
            run = require_one(completed_run_candidates(runs), f"{block} fold {fold}")
            entries.append(
                add_entry(
                    root,
                    block,
                    "DEEP-008",
                    block,
                    model,
                    config,
                    run,
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
        run = require_one(
            completed_run_candidates(runs), f"DEEP-008 weighted-F fold {fold}"
        )
        entries.append(
            add_entry(
                root,
                "DEEP-008_weighted-F",
                "DEEP-008",
                "weighted-F",
                "gated_fusion",
                config,
                run,
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
            run = require_one(
                completed_run_candidates(runs), f"Stage G {method} fold {fold}"
            )
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


def foldwise_mean_sd(values: list[float]) -> tuple[float, float]:
    if len(values) != 5:
        raise ValueError(f"expected five fold values, found {len(values)}")
    return statistics.mean(values), statistics.stdev(values)


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
                        mean, deviation = foldwise_mean_sd(values)
                        row[f"{corpus}_{metric}_mean"] = f"{mean:.10f}"
                        row[f"{corpus}_{metric}_sd"] = f"{deviation:.10f}"
                writer.writerow(row)


STAGE_I_FAMILIES = (
    "emotionmsp_embeds",
    "emotionmsp_trajectories",
    "trillsson4_1",
    "wav2vec_xlsr300m",
)
STAGE_II_SUBSETS = (
    "emotionmsp_embeds__emotionmsp_trajectories",
    "emotionmsp_embeds__emotionmsp_trajectories__trillsson4_1",
    "emotionmsp_embeds__emotionmsp_trajectories__trillsson4_1__wav2vec_xlsr300m",
    "emotionmsp_embeds__emotionmsp_trajectories__wav2vec_xlsr300m",
    "emotionmsp_embeds__trillsson4_1",
    "emotionmsp_embeds__trillsson4_1__wav2vec_xlsr300m",
    "emotionmsp_embeds__wav2vec_xlsr300m",
    "emotionmsp_trajectories__trillsson4_1",
    "emotionmsp_trajectories__trillsson4_1__wav2vec_xlsr300m",
    "emotionmsp_trajectories__wav2vec_xlsr300m",
    "trillsson4_1__wav2vec_xlsr300m",
)
STAGE_G_METHODS = (
    "equal",
    "stch",
    "progress_corrected",
    "ra_stch_corrected",
    "gradnorm_corrected",
    "pcgrad",
    "cagrad",
    "dbmtl",
)


def expected_blocks() -> set[str]:
    stage_i = {
        f"DEEP-003_{family}_{model}"
        for family in STAGE_I_FAMILIES
        for model in ("mlp", "mdformer")
    }
    stage_ii = {
        f"DEEP-006_{subset}_{model}"
        for subset in STAGE_II_SUBSETS
        for model in ("mlp", "mdformer")
    }
    return (
        stage_i
        | stage_ii
        | {
            "DEEP-008_A",
            "DEEP-008_F",
            "DEEP-008_weighted-F",
        }
        | {f"DEEP-008_StageG_{method}" for method in STAGE_G_METHODS}
    )


def validate_grid(entries: list[Entry]) -> None:
    actual_blocks = {entry.block for entry in entries}
    if actual_blocks != expected_blocks():
        raise ValueError(
            f"experimental block grid mismatch: missing={sorted(expected_blocks() - actual_blocks)} "
            f"unexpected={sorted(actual_blocks - expected_blocks())}"
        )
    if len(entries) != 205 or len(actual_blocks) != 41:
        raise ValueError(
            f"expected 205 entries/41 blocks, got {len(entries)}/{len(actual_blocks)}"
        )
    by_block = {}
    for entry in entries:
        by_block.setdefault(entry.block, []).append(entry)
    if any(
        sorted(item.fold for item in rows) != list(range(5))
        for rows in by_block.values()
    ):
        raise ValueError("every report block must contain folds 0 through 4")
    if (
        len({entry.mlflow_run_id for entry in entries}) != 200
        or len({entry.run_dir for entry in entries}) != 200
    ):
        raise ValueError("expected exactly 200 unique physical runs and MLflow run IDs")
    if (
        len(
            {
                entry.block
                for entry in entries
                if entry.block.startswith("DEEP-008_StageG_")
            }
        )
        != 8
    ):
        raise ValueError("expected eight Stage G methods")
    corrected = [
        entry
        for entry in entries
        if entry.configuration
        in {"progress_corrected", "ra_stch_corrected", "gradnorm_corrected"}
    ]
    if len(corrected) != 15 or not all(
        "stageg_wsm_corrected" in str(entry.run_dir) for entry in corrected
    ):
        raise ValueError("expected 15 corrected Stage G replacement runs")
    equal = {
        entry.fold: entry for entry in entries if entry.block == "DEEP-008_StageG_equal"
    }
    weighted = {
        entry.fold: entry for entry in entries if entry.block == "DEEP-008_weighted-F"
    }
    for fold in range(5):
        if (
            equal[fold].run_dir != weighted[fold].run_dir
            or equal[fold].checkpoint != weighted[fold].checkpoint
            or equal[fold].values != weighted[fold].values
            or equal[fold].mlflow_run_id != weighted[fold].mlflow_run_id
        ):
            raise ValueError(
                f"Stage G Equal does not exactly reuse weighted-F fold {fold}"
            )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "docs/evidence/tables"
    entries = historical_entries(root) + deep008_entries(root)
    validate_grid(entries)
    output.mkdir(parents=True, exist_ok=True)
    write_per_fold(entries, output / "DEEP008_HISTORICAL_TEST_METRICS_PER_FOLD.csv")
    write_aggregate(entries, output / "DEEP008_HISTORICAL_TEST_METRICS_AGGREGATED.csv")
    write_manifest(entries, output / "DEEP008_HISTORICAL_TEST_SOURCE_MANIFEST.csv")
    print("report_blocks=41 fold_entries=205 aggregate_rows=123")


if __name__ == "__main__":
    main()
