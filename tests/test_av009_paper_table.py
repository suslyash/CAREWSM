import csv
import math
from pathlib import Path

import pytest

from scripts.fusion.generate_paper_table import (
    _mean_sd,
    cell_values,
    generate,
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


def test_generated_outputs_have_eight_blocks_and_24_data_rows() -> None:
    root = Path(__file__).parents[1]
    selections = generate(root / "logs/fusion/sweep/reports", root / "reports/av_fusion_sweep")
    assert len(selections) == 8
    table = (root / "reports/av_fusion_sweep/FUSION_PAPER_TABLE.tex").read_text()
    assert table.count("& Depression &") == 8
    assert table.count("& PD &") == 8
    assert table.count("& Average &") == 8
    assert "TBD" not in table
    assert "\\pm" in table


def test_selection_csv_contains_five_fold_identities() -> None:
    root = Path(__file__).parents[1]
    path = root / "reports/av_fusion_sweep/FUSION_PAPER_TABLE_SELECTION.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert all(len(row["fold_identities"].split(",")) == 5 for row in rows)
    assert all(math.isfinite(float(row["val_overall_uar_mean"])) for row in rows)
