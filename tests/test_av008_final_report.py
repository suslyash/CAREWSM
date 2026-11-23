"""AV-008 report verification tests."""

import csv
from pathlib import Path

from scripts.fusion.final_report import (
    best_rows, verify, verify_ranking_and_guided,
)


def load(name):
    with (Path("logs/fusion/sweep/reports") / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_audited_matrix_has_complete_results_and_metric_cardinality():
    counts = verify(load("runs_status.csv"), load("results_summary.csv"))
    assert counts == {"runs": 285, "variants": 57, "metrics_per_run": 48}


def test_final_report_and_figures_are_regenerated():
    report = Path("reports/av_fusion_sweep/FINAL_RESEARCH_REPORT.md")
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "285/285 runs" in text
    assert "UPSTREAM_PROVENANCE=UNKNOWN" in text
    for name in (
        "val_overall_uar_top15.svg",
        "architecture_val_means.svg",
        "audio_minus_text_overall.svg",
    ):
        assert (Path("reports/av_fusion_sweep/figures") / name).is_file()



def test_five_fold_aggregates_are_recomputed_from_verified_rows():
    import pytest

    runs = load("runs_status.csv")
    summaries = load("results_summary.csv")
    target = dict(summaries[0])
    target["val_overall_uar_mean"] = str(
        float(target["val_overall_uar_mean"]) + 0.02
    )
    with pytest.raises(ValueError, match="five-fold aggregate mismatch"):
        verify(runs, [target, *summaries[1:]])


def test_val_ranking_and_paired_guided_differences_match_source_rows():
    import pytest

    runs = load("runs_status.csv")
    summaries = load("results_summary.csv")
    ranking = load("ranking_val.csv")
    guided = load("comparison_audio_vs_text_guided.csv")
    verify_ranking_and_guided(runs, summaries, ranking, guided)
    broken = dict(guided[0])
    broken["val_overall_uar_audio_minus_text"] = str(
        float(broken["val_overall_uar_audio_minus_text"]) + 0.05
    )
    with pytest.raises(ValueError, match="guided difference mismatch"):
        verify_ranking_and_guided(runs, summaries, ranking, [broken, *guided[1:]])
    with pytest.raises(ValueError, match="ranking does not follow"):
        verify_ranking_and_guided(runs, summaries, list(reversed(ranking)), guided)


def test_best_by_task_reports_the_corresponding_descriptive_test_metric():
    summaries = load("results_summary.csv")
    by_id = {row["experiment_id"]: row for row in summaries}
    output = best_rows(summaries)
    chosen = next(row for row in output if row["category"] == "care_depression")
    assert chosen["selection_metric"] == "val_care_depression_macro_recall_mean"
    assert chosen["test_selected_metric"] == "test_care_depression_macro_recall_mean"
    source = by_id[chosen["experiment_id"]]
    assert chosen["test_selected_metric_mean"] == source[
        "test_care_depression_macro_recall_mean"
    ]
    assert chosen["test_selected_metric_sd"] == source[
        "test_care_depression_macro_recall_sd"
    ]
