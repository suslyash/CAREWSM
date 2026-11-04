#!/usr/bin/env python3
"""Deterministically aggregate completed CAREWSM audio runs.

This is a read-only analysis tool: it only reads logs/audio/**/summary.csv and
writes tracked derived evidence tables/reports.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


TASKS = ("depression", "parkinson")
METRICS = (
    "macro_f1",
    "weighted_f1",
    "macro_recall",
    "weighted_recall",
    "patient_recall",
    "healthy_recall",
)
SPLITS = ("VAL CARE", "VAL WSM", "VAL average", "TEST CARE", "TEST WSM")
EXPECTED_GROUPS = 22
EXPECTED_FOLDS = 110
KNOWN_DUPLICATE = ("disvoice_prosody_dynamic", "mptsnet", 3)


@dataclass(frozen=True)
class Run:
    family: str
    model: str
    fold: int
    path: Path
    epoch: int
    monitor: float
    values: dict[tuple[str, str, str], float]


def parse_key(path: Path, root: Path) -> tuple[str, str, int]:
    family, model, run_name = path.relative_to(root).parts[:3]
    match = re.search(r"(?:fold|mlp_fold)(\d+)_", run_name)
    if match is None:
        raise ValueError(f"cannot parse fold from {path}")
    return family, model, int(match.group(1))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"epoch", "split", "task", *METRICS}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return rows


def select_epoch(rows: list[dict[str, str]], path: Path) -> tuple[int, float]:
    epochs = sorted({int(row["epoch"]) for row in rows})
    candidates: list[tuple[float, int]] = []
    for epoch in epochs:
        selected = [
            row
            for row in rows
            if int(row["epoch"]) == epoch and row["split"] == "VAL average"
        ]
        if {row["task"] for row in selected} != set(TASKS):
            raise ValueError(f"{path}: VAL average incomplete at epoch {epoch}")
        monitor = statistics.mean(float(row["macro_f1"]) for row in selected)
        candidates.append((monitor, epoch))
    if not candidates:
        raise ValueError(f"{path}: no VAL average rows")
    # max monitor; min epoch makes exact ties deterministic and validation-only.
    monitor, epoch = max(candidates, key=lambda item: (item[0], -item[1]))
    return epoch, monitor


def load_run(summary: Path, root: Path) -> Run:
    rows = read_rows(summary)
    epoch, monitor = select_epoch(rows, summary)
    values: dict[tuple[str, str, str], float] = {}
    for split in SPLITS:
        for task in TASKS:
            matches = [
                row
                for row in rows
                if int(row["epoch"]) == epoch
                and row["split"] == split
                and row["task"] == task
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{summary}: expected one {split}/{task} row at epoch {epoch}, "
                    f"found {len(matches)}"
                )
            for metric in METRICS:
                value = float(matches[0][metric])
                if value != value or value in (float("inf"), float("-inf")):
                    raise ValueError(f"{summary}: non-finite {split}/{task}/{metric}")
                values[(split, task, metric)] = value
    family, model, fold = parse_key(summary, root)
    return Run(family, model, fold, summary.parent, epoch, monitor, values)


def discover(root: Path) -> tuple[list[Run], list[Path]]:
    summaries = sorted(root.glob("**/summary.csv"))
    grouped: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    for summary in summaries:
        grouped[parse_key(summary, root)].append(summary)
    duplicate_keys = {key for key, paths in grouped.items() if len(paths) > 1}
    if duplicate_keys != {KNOWN_DUPLICATE}:
        raise ValueError(f"unexpected duplicate canonical keys: {sorted(duplicate_keys)}")
    selected: list[Run] = []
    duplicates: list[Path] = []
    for key, paths in sorted(grouped.items()):
        paths = sorted(paths, key=lambda path: path.parent.name)
        if key == KNOWN_DUPLICATE:
            duplicates.extend(paths[:-1])
            paths = paths[-1:]
        for path in paths:
            selected.append(load_run(path, root))
    if len(selected) != EXPECTED_FOLDS:
        raise ValueError(f"expected {EXPECTED_FOLDS} canonical runs, found {len(selected)}")
    by_group: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in selected:
        by_group[(run.family, run.model)].append(run)
    if len(by_group) != EXPECTED_GROUPS:
        raise ValueError(f"expected {EXPECTED_GROUPS} groups, found {len(by_group)}")
    if any(sorted(run.fold for run in runs) != list(range(5)) for runs in by_group.values()):
        raise ValueError("every canonical group must contain folds 0, 1, 2, 3, 4 exactly once")
    return sorted(selected, key=lambda run: (run.family, run.model, run.fold)), duplicates


def mean_std(values: list[float]) -> str:
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values) * 100:.2f} +/- {deviation * 100:.2f}"


def aggregate(runs: list[Run], split: str, task: str, metric: str) -> str:
    return mean_std([run.values[(split, task, metric)] for run in runs])


def average_block(runs: list[Run], task: str, metric: str, prefix: str) -> str:
    values = []
    for run in runs:
        values.append(
            statistics.mean(
                [
                    run.values[(f"{prefix} CARE", task, metric)],
                    run.values[(f"{prefix} WSM", task, metric)],
                ]
            )
        )
    return mean_std(values)


def write_fold_csv(runs: list[Run], output: Path) -> None:
    fields = ["configuration", "feature", "model", "fold", "selected_epoch", "val_monitor"]
    for phase in ("VAL", "TEST"):
        for corpus in ("CARE", "WSM", "Average"):
            for task in TASKS:
                for metric in METRICS:
                    fields.append(f"{phase}_{corpus}_{task}_{metric}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {
                "configuration": f"{run.family}/{run.model}",
                "feature": run.family,
                "model": run.model,
                "fold": run.fold,
                "selected_epoch": run.epoch,
                "val_monitor": f"{run.monitor:.8f}",
            }
            for phase in ("VAL", "TEST"):
                prefix = phase
                for corpus in ("CARE", "WSM"):
                    split = f"{prefix} {corpus}"
                    for task in TASKS:
                        for metric in METRICS:
                            row[f"{phase}_{corpus}_{task}_{metric}"] = run.values[
                                (split, task, metric)
                            ]
                for task in TASKS:
                    for metric in METRICS:
                        row[f"{phase}_Average_{task}_{metric}"] = statistics.mean(
                            [
                                run.values[(f"{prefix} CARE", task, metric)],
                                run.values[(f"{prefix} WSM", task, metric)],
                            ]
                        )
            writer.writerow(row)


def ranking(groups: dict[tuple[str, str], list[Run]]) -> list[tuple[float, tuple[str, str], list[Run]]]:
    result = []
    for key, runs in groups.items():
        result.append((statistics.mean([run.monitor for run in runs]), key, runs))
    return sorted(result, reverse=True, key=lambda item: (item[0], item[1]))


def report(runs: list[Run], duplicates: list[Path], output: Path) -> None:
    groups: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in runs:
        groups[(run.family, run.model)].append(run)
    val_rank = ranking(groups)
    test_rank = sorted(
        val_rank,
        key=lambda item: statistics.mean(
            [
                statistics.mean(
                    [
                        statistics.mean(
                            [
                                run.values[(f"TEST {corpus}", task, "macro_f1")]
                                for corpus in ("CARE", "WSM")
                            ]
                        )
                        for run in item[2]
                    ]
                )
                for task in TASKS
            ]
        ),
        reverse=True,
    )
    lines = [
        "# CAREWSM audio experiment report",
        "",
        "Date: 2026-11-04  ",
        "Branch: `codex/manager-report-2026-11-04`  ",
        "",
        "## Executive summary",
        "",
        "This report is validation-first. Every canonical fold-run selects one epoch using only the maximum `joint/average_macro_f1`, represented in `summary.csv` by the mean of the two `VAL average` task Macro F1 rows at that epoch. Exact ties use the earliest epoch. VAL and TEST metrics below come from that same epoch.",
        "",
        f"- Canonical configurations: **{len(groups)}/22**.",
        f"- Canonical fold-runs: **{len(runs)}/110**.",
        f"- Known historical duplicate excluded deterministically: **{len(duplicates)}**.",
        "- Mean +/- std is across five folds; Average is the arithmetic CARE/WSM mean, never pooled predictions.",
        "",
        "## Why the matrix has 22 configurations",
        "",
        "The matrix has 11 feature families with five folds each. Static/T=1 families use MLP + MD-Former; temporal families use MPTSNet + MD-Former. TSCMamba is absent because its `mamba_ssm` dependency is blocked/unavailable; it was not experimentally evaluated and lost.",
        "",
        "## Ranking by VAL joint/average_macro_f1",
        "",
        "This is the primary ranking for feature/model decisions. TEST is not used.",
        "",
        "| Rank | Configuration | Mean VAL monitor | VAL std | Mean best epoch | Valid folds | Collapse warning/count |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for index, (score, (family, model), group) in enumerate(val_rank, 1):
        collapse = sum(
            run.values[("VAL average", task, "patient_recall")] <= 0.02
            and run.values[("VAL average", task, "healthy_recall")] >= 0.98
            for run in group
            for task in TASKS
        )
        monitor_std = statistics.stdev([run.monitor for run in group])
        lines.append(
            f"| {index} | `{family}/{model}` | {score:.4f} | {monitor_std:.4f} | "
            f"{statistics.mean([run.epoch for run in group]):.1f} | {len(group)} | {collapse} |")
    lines += [
        "",
        "## Full aggregate VAL results",
        "",
        "All six metrics are shown for CARE, WSM and their arithmetic Average at the VAL-selected epoch.",
        "",
    ]
    lines.extend(full_table(groups, "VAL"))
    lines += [
        "",
        "## Full aggregate TEST results",
        "",
        "TEST ranking is statistics only and was not used for selection or promotion.",
        "",
    ]
    lines.extend(full_table(groups, "TEST"))
    lines += [
        "",
        "## Descriptive TEST ranking",
        "",
        "TEST ranking is statistics only and was not used for selection or promotion.",
        "",
        "| Rank | Configuration | TEST Average Macro F1 |",
        "|---:|---|---:|",
    ]
    for index, (_, key, group) in enumerate(test_rank, 1):
        score = statistics.mean(
            [
                statistics.mean(
                    [
                        statistics.mean(
                            [run.values[(f"TEST {corpus}", task, "macro_f1")] for corpus in ("CARE", "WSM")]
                        )
                        for run in group
                    ]
                )
                for task in TASKS
            ]
        )
        lines.append(f"| {index} | `{key[0]}/{key[1]}` | {score * 100:.2f} |")
    lines += [
        "",
        "## Collapse findings",
        "",
        "A collapse warning is raised when VAL Average Patient Recall is <= 2% and Healthy Recall is >= 98% for a task/fold. Weighted metrics must not hide this failure mode.",
        "",
    ]
    for _, (family, model), group in val_rank:
        warnings = [
            f"fold {run.fold} {task}"
            for run in group
            for task in TASKS
            if run.values[("VAL average", task, "patient_recall")] <= 0.02
            and run.values[("VAL average", task, "healthy_recall")] >= 0.98
        ]
        if warnings:
            lines.append(f"- `{family}/{model}`: {', '.join(warnings)}")
    lines += [
        "",
        "## Reproducibility and integrity",
        "",
        "- Source: `logs/audio/**/summary.csv`; no raw logs, checkpoints or MLflow artifacts are tracked.",
        "- Fold-level derived evidence: `docs/evidence/MANAGER_AUDIO_EXPERIMENT_FOLD_RESULTS_2026-11-04.csv`.",
        "- Reproducer: `scripts/analyze_audio_experiments.py`.",
        "- The script fails on missing metrics, unexpected duplicates, non-finite values, incomplete folds or ambiguous rows.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def full_table(groups: dict[tuple[str, str], list[Run]], phase: str) -> list[str]:
    lines = [
        "| Configuration | Task | CARE Macro F1 | CARE Weighted F1 | CARE Macro Recall | CARE Weighted Recall | CARE Patient Recall | CARE Healthy Recall | WSM Macro F1 | WSM Weighted F1 | WSM Macro Recall | WSM Weighted Recall | WSM Patient Recall | WSM Healthy Recall | Average Macro F1 | Average Weighted F1 | Average Macro Recall | Average Weighted Recall | Average Patient Recall | Average Healthy Recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (family, model), group in sorted(groups.items()):
        for task in TASKS:
            values = []
            for metric in METRICS:
                values.append(aggregate(group, f"{phase} CARE", task, metric))
            for metric in METRICS:
                values.append(aggregate(group, f"{phase} WSM", task, metric))
            for metric in METRICS:
                values.append(average_block(group, task, metric, phase))
            label = "Depression" if task == "depression" else "PD"
            lines.append(f"| `{family}/{model}` | {label} | " + " | ".join(values) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=Path("logs/audio"))
    parser.add_argument("--report", type=Path, default=Path("docs/evidence/MANAGER_AUDIO_EXPERIMENT_REPORT_2026-11-04.md"))
    parser.add_argument("--fold-csv", type=Path, default=Path("docs/evidence/MANAGER_AUDIO_EXPERIMENT_FOLD_RESULTS_2026-11-04.csv"))
    args = parser.parse_args()
    runs, duplicates = discover(args.logs)
    write_fold_csv(runs, args.fold_csv)
    report(runs, duplicates, args.report)
    print(f"canonical_groups=22 canonical_runs={len(runs)} excluded_duplicates={len(duplicates)}")
    print(f"report={args.report}")
    print(f"fold_csv={args.fold_csv}")


if __name__ == "__main__":
    main()
