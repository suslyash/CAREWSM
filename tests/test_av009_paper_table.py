import csv
import math
import re
from pathlib import Path

import pytest

from scripts.fusion.generate_paper_table import (
    _mean_sd,
    cell_values,
    compact_feature_ids,
    generate,
    highlight_extrema,
    select_top_configurations,
)


def test_sample_sd_is_computed_from_fold_level_values() -> None:
    result = _mean_sd([0.10, 0.20, 0.40, 0.70, 0.90])
    assert result.mean == pytest.approx(0.46)
    assert result.sd == pytest.approx(0.3361547263)
    assert result.sd != pytest.approx(sum([0.1, 0.2, 0.3, 0.4, 0.5]) / 5)


def test_fold_level_average_is_aggregated_before_sd() -> None:
    rows = []
    for fold in range(5):
        row = {"experiment_id": "x", "fold": str(fold)}
        for corpus, base in (("care", 0.1), ("wsm", 0.5)):
            for task, offset in (("depression", 0), ("parkinson", 0.2)):
                for metric in ("macro_recall", "patient_recall", "healthy_recall"):
                    row[f"test_{corpus}_{task}_{metric}"] = str(base + offset + fold * 0.01)
        rows.append(row)
    values = cell_values(rows, "test", "depression", "macro_recall")
    assert values == pytest.approx([0.3, 0.31, 0.32, 0.33, 0.34])
    assert _mean_sd(values).sd == pytest.approx(0.0158113883)


def test_generated_outputs_have_twelve_blocks_and_36_data_rows() -> None:
    root = Path(__file__).parents[1]
    selections = generate(root / "logs/fusion/sweep/reports", root / "reports/av_fusion_sweep")
    assert len(selections) == 12
    table = (root / "reports/av_fusion_sweep/FUSION_PAPER_TABLE.tex").read_text()
    assert table.count("& Depression &") == 12
    assert table.count("& PD &") == 12
    assert table.count("& Average &") == 12
    data_lines = [line for line in table.splitlines() if any(f"& {task} &" in line for task in ("Depression", "PD", "Average"))]
    assert len(data_lines) == 36
    assert sum(line.count(r"\pm") for line in data_lines) == 324
    assert "TBD" not in table
    assert "undefined" not in table
    assert "PREVIEW ONLY" not in table
    assert "\\pm" in table
    assert r"\multirow{2}{*}{\shortstack[l]{Model /\\feature IDs}}" in table
    assert table.count(r"\shortstack{Macro\\Recall}") == 3
    assert table.count(r"\shortstack{Patient\\Recall}") == 3
    assert table.count(r"\shortstack{Healthy\\Recall}") == 3
    assert " & Modalities & " not in table
    assert r"\multirow{3}{*}{\raggedright\shortstack[l]{Shared MTL,\\\texttt{A1V4}}}" in table
    assert r"\multirow{3}{*}{\raggedright\shortstack[l]{Cross-Attention,\\\texttt{A2V4T9}}}" in table
    assert table.count(r"\multirow{3}{*}") == 12
    assert r"\cline{3-11}" in table
    assert table.index(r"\caption{") < table.index(r"\resizebox{")
    assert table.index(r"\label{tab:fusion-paper}") < table.index(r"\resizebox{")
    assert table.count(r"\hline") >= 13
    assert table.count(r"\begin{tabular}{|p{2.85cm}|p{1.75cm}|r|r|r|r|r|r|r|r|r|}") == 1
    assert table.count(r"\multicolumn{11}{|c|}{\textbf{Shared MTL -- Best by VAL}}") == 1
    assert table.count(r"\multicolumn{11}{|c|}{\textbf{Cross-Attention -- Best by TEST (post-hoc)}}") == 1
    assert r"\color{blue}" in table and r"\color{red}" in table
    assert not re.search(r"\{\\color\{(?:red|blue)\}(?:0\.00|100\.00)\}", table)


def test_selection_csv_contains_five_fold_identities() -> None:
    root = Path(__file__).parents[1]
    path = root / "reports/av_fusion_sweep/FUSION_PAPER_TABLE_SELECTION.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert sum(row["architecture"] == "cross_attention" for row in rows) == 6
    for selection in ("Best by VAL", "Best by TEST (post-hoc)"):
        cross = [
            row for row in rows
            if row["architecture"] == "cross_attention"
            and row["selection_criterion"] == selection
        ]
        assert len(cross) == 3
        assert sorted(int(row["selection_rank"]) for row in cross) == [1, 2, 3]
        assert len({row["experiment_id"] for row in cross}) == 3
    assert all(len(row["fold_identities"].split(",")) == 5 for row in rows)
    assert all(math.isfinite(float(row["val_overall_uar_mean"])) for row in rows)



def test_compact_model_features_use_exact_original_modality_ids() -> None:
    assert compact_feature_ids("audio_id1+video_4") == "A1V4"
    assert compact_feature_ids("audio_id2+text_id10") == "A2T10"
    assert compact_feature_ids("video_4+text_id9") == "V4T9"
    assert compact_feature_ids("audio_id2+video_4+text_id9") == "A2V4T9"
    with pytest.raises(ValueError, match="noncanonical"):
        compact_feature_ids("text_id9+audio_id1")


def test_extrema_colors_per_task_and_metric_column_skip_boundary_means() -> None:
    def entry(task: str, means: list[float]) -> dict[str, object]:
        return {
            "task": task,
            "cells": ["$" + f"{value:.2f}" + r" \pm 1.00$" for value in means],
        }

    rows = [
        entry("depression", [0.0, 40.0] + [0.0] * 7),
        entry("depression", [100.0, 70.0] + [100.0] * 7),
        entry("depression", [40.0, 70.0] + [40.0] * 7),
        entry("depression", [70.0, 60.0] + [70.0] * 7),
        entry("parkinson", [55.0] * 9),
    ]
    highlight_extrema(rows)
    assert rows[0]["cells"][0].startswith("$0.00")
    assert rows[1]["cells"][0].startswith("$100.00")
    assert rows[2]["cells"][0].startswith("$" + r"{\color{red}40.00}")
    assert rows[3]["cells"][0].startswith("$" + r"{\color{blue}70.00}")
    assert rows[0]["cells"][1].startswith("$" + r"{\color{red}40.00}")
    assert rows[1]["cells"][1].startswith("$" + r"{\color{blue}70.00}")
    assert rows[2]["cells"][1].startswith("$" + r"{\color{blue}70.00}")
    assert r"\color{" not in rows[4]["cells"][0]
    assert rows[1]["cells"][1].endswith(r"\pm 1.00$")



def test_cross_attention_selects_three_distinct_top_configurations() -> None:
    rows = []
    for features, val_score, test_score in (
        ("A1_V4_T9", 0.70, 0.69),
        ("A2_V4_T9", 0.72, 0.68),
        ("A4_V4_T9", 0.68, 0.72),
        ("A1_V4_T10", 0.73, 0.67),
    ):
        for fold in range(5):
            rows.append({
                "experiment_id": "cross_attention__" + features,
                "architecture": "cross_attention",
                "modalities": "audio+video+text",
                "fold": str(fold),
                "val_overall_uar": str(val_score),
                "test_overall_uar": str(test_score),
            })
    val = select_top_configurations(
        rows, "cross_attention", "audio+video+text", "val", 3
    )
    test = select_top_configurations(
        rows, "cross_attention", "audio+video+text", "test", 3
    )
    assert [experiment for experiment, _ in val] == [
        "cross_attention__A1_V4_T10",
        "cross_attention__A2_V4_T9",
        "cross_attention__A1_V4_T9",
    ]
    assert [experiment for experiment, _ in test] == [
        "cross_attention__A4_V4_T9",
        "cross_attention__A1_V4_T9",
        "cross_attention__A2_V4_T9",
    ]
