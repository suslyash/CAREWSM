from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest

SPEC = spec_from_file_location(
    "export_historical_test_metrics",
    Path(__file__).parents[1] / "scripts/export_historical_test_metrics.py",
)
EXPORTER = module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORTER
SPEC.loader.exec_module(EXPORTER)


def write_summary(path, rows):
    path.write_text(
        "epoch;dev_mean_macro_recall;test_mean_macro_recall\n"
        + "\n".join(f"{epoch};{val};{test}" for epoch, val, test in rows)
        + "\n",
        encoding="utf-8",
    )


def test_selects_validation_epoch_not_test_best(tmp_path):
    summary = tmp_path / "summary.txt"
    write_summary(summary, [(1, 0.60, 0.90), (2, 0.80, 0.10)])
    _, epoch, monitor = EXPORTER.summary_row(summary)
    assert (epoch, monitor) == (2, 0.80)


def test_equal_validation_scores_select_earliest_epoch(tmp_path):
    summary = tmp_path / "summary.txt"
    write_summary(summary, [(1, 0.80, 0.10), (2, 0.80, 0.90)])
    _, epoch, _ = EXPORTER.summary_row(summary)
    assert epoch == 1


def test_foldwise_equal_corpus_task_uses_sample_sd():
    mean, deviation = EXPORTER.foldwise_mean_sd([0.0, 1.0, 2.0, 3.0, 4.0])
    assert mean == 2.0
    assert deviation == pytest.approx((2.5) ** 0.5)


def test_fold_first_sd_differs_from_component_sd_average():
    fold_values = [(a + b) / 2 for a, b in zip([0, 0, 0, 0, 1], [0, 0, 0, 1, 0])]
    _, fold_sd = EXPORTER.foldwise_mean_sd(fold_values)
    component_sd_average = (
        EXPORTER.foldwise_mean_sd([0, 0, 0, 0, 1])[1]
        + EXPORTER.foldwise_mean_sd([0, 0, 0, 1, 0])[1]
    ) / 2
    assert fold_sd != component_sd_average


def test_ambiguous_runtime_directories_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        EXPORTER.require_one([tmp_path / "run_a", tmp_path / "run_b"], "synthetic")
