#!/usr/bin/env python3
"""Generate the AV-009 publication table from audited fusion fold results."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


METRICS = ("macro_recall", "patient_recall", "healthy_recall")
TASKS = ("depression", "parkinson")
CORPORA = ("care", "wsm")
ARCHITECTURES = {
    "bimodal_shared_mtl": (
        ("audio+video", "Audio + Video"),
        ("audio+text", "Audio + Text"),
        ("video+text", "Video + Text"),
    ),
    "cross_attention": (("audio+video+text", "Audio + Video + Text"),),
}


MODEL_NAMES = {
    "bimodal_shared_mtl": "Shared MTL",
    "cross_attention": "Cross-Attention",
}
SELECTION_ORDER = ("Best by VAL", "Best by TEST (post-hoc)")


def compact_feature_ids(feature_ids: str) -> str:
    """Compact and validate feature IDs, e.g. A2V4T9."""
    prefixes = {"audio_id": ("A", 0), "video_": ("V", 1), "text_id": ("T", 2)}
    pieces = []
    positions = []
    for token in feature_ids.split("+"):
        match = re.fullmatch(r"(audio_id|video_|text_id)(\d+)", token)
        if match is None:
            raise ValueError(f"unsupported feature ID: {token}")
        letter, position = prefixes[match.group(1)]
        pieces.append(f"{letter}{match.group(2)}")
        positions.append(position)
    if positions != sorted(set(positions)) or len(positions) not in (2, 3):
        raise ValueError(f"noncanonical modality combination: {feature_ids}")
    return "".join(pieces)

@dataclass(frozen=True)
class FoldSummary:
    mean: float
    sd: float


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{key} is not finite or is outside [0, 1]: {value}")
    return value


def _mean_sd(values: Iterable[float]) -> FoldSummary:
    values = list(values)
    if len(values) != 5:
        raise ValueError(f"expected five values, got {len(values)}")
    return FoldSummary(statistics.mean(values), statistics.stdev(values))


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 285:
        raise ValueError(f"expected 285 fold rows, got {len(rows)}")
    return rows


def validate_rows(rows: list[dict[str, str]]) -> None:
    identities = [(row["experiment_id"], row["fold"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicated experiment/fold identity")
    for row in rows:
        if row["status"] != "COMPLETED" or row["launcher_exit_code"] != "0":
            raise ValueError(f"incomplete row: {row['run_name']}")
        for corpus in CORPORA:
            for task in TASKS:
                for metric in METRICS:
                    _float(row, f"val_{corpus}_{task}_{metric}")
                    _float(row, f"test_{corpus}_{task}_{metric}")
        _float(row, "val_overall_uar")
        _float(row, "test_overall_uar")


def configuration_rows(
    rows: list[dict[str, str]], architecture: str, modalities: str
) -> dict[str, list[dict[str, str]]]:
    selected = [
        row
        for row in rows
        if row["architecture"] == architecture and row["modalities"] == modalities
    ]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        grouped.setdefault(row["experiment_id"], []).append(row)
    for experiment_id, group in grouped.items():
        folds = sorted(int(row["fold"]) for row in group)
        if folds != [0, 1, 2, 3, 4]:
            raise ValueError(f"{experiment_id} does not have folds 0-4: {folds}")
    return grouped


def overall_score(group: list[dict[str, str]], split: str) -> FoldSummary:
    return _mean_sd(float(row[f"{split}_overall_uar"]) for row in group)


def select_configuration(
    rows: list[dict[str, str]], architecture: str, modalities: str, split: str
) -> tuple[str, list[dict[str, str]]]:
    grouped = configuration_rows(rows, architecture, modalities)
    if not grouped:
        raise ValueError(f"no rows for {architecture}/{modalities}")
    experiment_id = max(
        grouped,
        key=lambda key: (overall_score(grouped[key], split).mean, key),
    )
    return experiment_id, grouped[experiment_id]


def check_summary_and_ranking(
    rows: list[dict[str, str]], summary_path: Path, ranking_path: Path
) -> None:
    with summary_path.open(newline="") as handle:
        summary = {row["experiment_id"]: row for row in csv.DictReader(handle)}
    with ranking_path.open(newline="") as handle:
        ranking = {row["experiment_id"]: row for row in csv.DictReader(handle)}
    for experiment_id, group in _group_by_id(rows).items():
        if experiment_id not in summary or experiment_id not in ranking:
            raise ValueError(f"missing audited summary/ranking row: {experiment_id}")
        for split in ("val", "test"):
            actual = overall_score(group, split)
            for suffix, value in (("mean", actual.mean), ("sd", actual.sd)):
                key = f"{split}_overall_uar_{suffix}"
                if not math.isclose(float(summary[experiment_id][key]), value, abs_tol=1e-9):
                    raise ValueError(f"summary mismatch for {experiment_id}/{key}")
                if not math.isclose(float(ranking[experiment_id][key]), value, abs_tol=1e-9):
                    raise ValueError(f"ranking mismatch for {experiment_id}/{key}")


def _group_by_id(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["experiment_id"], []).append(row)
    return groups


def validate_macro_recall(rows: list[dict[str, str]]) -> None:
    for row in rows:
        for split in ("val", "test"):
            for corpus in CORPORA:
                for task in TASKS:
                    macro = _float(row, f"{split}_{corpus}_{task}_macro_recall")
                    patient = _float(row, f"{split}_{corpus}_{task}_patient_recall")
                    healthy = _float(row, f"{split}_{corpus}_{task}_healthy_recall")
                    if not math.isclose(macro, (patient + healthy) / 2, abs_tol=1.1e-4):
                        raise ValueError(f"macro recall mismatch in {row['run_name']}")


def metric_fold_values(group: list[dict[str, str]], split: str, task: str, metric: str) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for corpus in CORPORA:
        values[corpus] = [
            _float(row, f"{split}_{corpus}_{task}_{metric}") for row in group
        ]
    return values


def cell_values(group: list[dict[str, str]], split: str, task: str, metric: str) -> list[float]:
    depression = metric_fold_values(group, split, "depression", metric)
    parkinson = metric_fold_values(group, split, "parkinson", metric)
    if task == "depression":
        return [(a + b) / 2 for a, b in zip(depression["care"], depression["wsm"])]
    if task == "parkinson":
        return [(a + b) / 2 for a, b in zip(parkinson["care"], parkinson["wsm"])]
    return [
        (cd + cp + wd + wp) / 4
        for cd, cp, wd, wp in zip(
            depression["care"], parkinson["care"], depression["wsm"], parkinson["wsm"]
        )
    ]


def format_cell(values: list[float]) -> str:
    result = _mean_sd(values)
    return f"${result.mean * 100:.2f} \\pm {result.sd * 100:.2f}$"


def metric_row(group: list[dict[str, str]], task: str) -> list[str]:
    care_cells: list[str] = []
    wsm_cells: list[str] = []
    average_cells: list[str] = []
    for metric in METRICS:
        depression = metric_fold_values(group, "test", "depression", metric)
        parkinson = metric_fold_values(group, "test", "parkinson", metric)
        if task == "average":
            care = [(a + b) / 2 for a, b in zip(depression["care"], parkinson["care"])]
            wsm = [(a + b) / 2 for a, b in zip(depression["wsm"], parkinson["wsm"])]
        else:
            source = metric_fold_values(group, "test", task, metric)
            care = source["care"]
            wsm = source["wsm"]
        care_cells.append(format_cell(care))
        wsm_cells.append(format_cell(wsm))
        average_cells.append(format_cell(cell_values(group, "test", task, metric)))
    return care_cells + wsm_cells + average_cells


def write_selection(path: Path, selections: list[dict[str, object]]) -> None:
    fields = [
        "architecture", "modality_combination", "selection_criterion", "experiment_id",
        "feature_ids", "val_overall_uar_mean", "val_overall_uar_sd",
        "test_overall_uar_mean", "test_overall_uar_sd", "fold_identities",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selections)


def highlight_extrema(rendered_rows: list[dict[str, object]]) -> None:
    """Highlight displayed non-boundary TEST means within each task and metric column.

    Compare all eight shown configurations (including tied selections). A rounded
    displayed 0.00% or 100.00% is excluded from extrema; SD remains uncolored.
    """
    for task in ("depression", "parkinson", "average"):
        matching = [row for row in rendered_rows if row["task"] == task]
        for column in range(9):
            candidates = []
            for row in matching:
                cell = row["cells"][column]
                match = re.fullmatch(r"\$([0-9]+\.[0-9]{2}) \\pm ([0-9]+\.[0-9]{2})\$", cell)
                if match is None:
                    raise ValueError(f"unexpected LaTeX metric cell: {cell}")
                value = float(match.group(1))
                if 0.00 < value < 100.00:
                    candidates.append((row, value))
            values = {value for _, value in candidates}
            if len(values) < 2:
                continue
            minimum, maximum = min(values), max(values)
            for row, value in candidates:
                if value not in (minimum, maximum):
                    continue
                color = "blue" if value == maximum else "red"
                colored_mean = "$" + "{\\color{" + color + "}" + f"{value:.2f}" + "}"
                row["cells"][column] = row["cells"][column].replace(
                    "$" + f"{value:.2f}", colored_mean, 1
                )


def write_latex(path: Path, blocks: list[dict[str, object]]) -> None:
    # Four coherent sections: selection modes grouped within each architecture.
    order = {
        (architecture, criterion, modalities): (arch_idx, selection_idx, modality_idx)
        for arch_idx, (architecture, modality_options) in enumerate(ARCHITECTURES.items())
        for selection_idx, criterion in enumerate(SELECTION_ORDER)
        for modality_idx, (modalities, _) in enumerate(modality_options)
    }
    display_blocks = sorted(
        blocks,
        key=lambda block: order[
            (block["architecture"], block["selection_criterion"], block["modalities"])
        ],
    )
    rendered_rows: list[dict[str, object]] = []
    for block in display_blocks:
        for task in ("depression", "parkinson", "average"):
            rendered_rows.append({
                "block": block, "task": task, "cells": metric_row(block["rows"], task),
            })
    highlight_extrema(rendered_rows)

    lines = [
        r"% Generated by scripts/fusion/generate_paper_table.py; requires booktabs, multirow, graphicx, xcolor.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lll*{9}{c}}",
        r"\toprule",
        r"Model, feature IDs & Modalities & Task & \multicolumn{3}{c}{CARE} & \multicolumn{3}{c}{WSM} & \multicolumn{3}{c}{CARE--WSM average} \\",
        r"& & & Macro Recall & Patient Recall & Healthy Recall & Macro Recall & Patient Recall & Healthy Recall & Macro Recall & Patient Recall & Healthy Recall \\",
        r"\midrule",
    ]
    current_section = None
    for row in rendered_rows:
        block = row["block"]
        task = row["task"]
        section = MODEL_NAMES[block["architecture"]] + " -- " + block["selection_criterion"]
        if section != current_section:
            current_section = section
            lines.extend([
                rf"\multicolumn{{12}}{{l}}{{\textbf{{{section}}}}} \\",
                r"\midrule",
            ])
        if task == "depression":
            experiment_id = str(block["experiment_id"])
            lines.append(rf"% provenance: {experiment_id}; folds 0,1,2,3,4")
            model = MODEL_NAMES[block["architecture"]]
            short_id = compact_feature_ids(str(block["feature_ids"]))
            first_column = model + r", \texttt{" + short_id + "}"
            modalities = str(block["modality_label"])
            prefix = (
                r"\multirow{3}{*}{" + first_column + r"} & "
                + r"\multirow{3}{*}{" + modalities + r"} & "
            )
        else:
            prefix = "& & "
        label = {"depression": "Depression", "parkinson": "PD", "average": "Average"}[task]
        lines.append(prefix + label + " & " + " & ".join(row["cells"]) + r" \\")
        if task == "average":
            lines.append(r"\cline{1-12}")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{Fusion comparison on the verified five-fold exploratory sweep. Entries are five-fold arithmetic mean $\pm$ sample standard deviation (ddof=1), in percent. A = audio, V = video, T = text; numerical suffixes identify feature versions. Best-by-VAL blocks select features using validation Overall UAR; best-by-TEST blocks are retrospective post-hoc descriptive selections and are not valid generalization estimates. Blue (red) highlights the highest (lowest) displayed non-boundary mean for each task and metric column across the eight configuration blocks, including ties. Displayed 0.00\% and 100.00\% means are never highlighted; SDs are not colored. All averages are derived at fold level before aggregation. $\texttt{UPSTREAM\_PROVENANCE=UNKNOWN -- EXPLORATORY RESULTS ONLY}$.}",
        r"\label{tab:fusion-paper}",
        r"\end{table*}",
    ])
    path.write_text("\n".join(lines) + "\n")


def generate(input_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    rows = load_rows(input_dir / "results_by_fold.csv")
    validate_rows(rows)
    validate_macro_recall(rows)
    check_summary_and_ranking(rows, input_dir / "results_summary.csv", input_dir / "ranking_val.csv")
    blocks: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    block_number = 1
    for architecture, modality_options in ARCHITECTURES.items():
        for modalities, modality_label in modality_options:
            for criterion, split in (("Best by VAL", "val"), ("Best by TEST (post-hoc)", "test")):
                experiment_id, group = select_configuration(rows, architecture, modalities, split)
                summary = {
                    "architecture": architecture,
                    "modality_combination": modality_label,
                    "selection_criterion": criterion,
                    "experiment_id": experiment_id,
                    "feature_ids": group[0]["feature_ids"],
                    "val_overall_uar_mean": f"{overall_score(group, 'val').mean:.6f}",
                    "val_overall_uar_sd": f"{overall_score(group, 'val').sd:.6f}",
                    "test_overall_uar_mean": f"{overall_score(group, 'test').mean:.6f}",
                    "test_overall_uar_sd": f"{overall_score(group, 'test').sd:.6f}",
                    "fold_identities": ",".join(row["run_name"] for row in sorted(group, key=lambda r: int(r["fold"]))),
                }
                selections.append(summary)
                blocks.append({
                    "id": f"{block_number}",
                    "architecture": architecture,
                    "modalities": modalities,
                    "modality_label": modality_label,
                    "selection_criterion": criterion,
                    "experiment_id": experiment_id,
                    "feature_ids": group[0]["feature_ids"],
                    "rows": group,
                })
                block_number += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    write_selection(output_dir / "FUSION_PAPER_TABLE_SELECTION.csv", selections)
    write_latex(output_dir / "FUSION_PAPER_TABLE.tex", blocks)
    return selections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("logs/fusion/sweep/reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/av_fusion_sweep"))
    args = parser.parse_args()
    generate(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
