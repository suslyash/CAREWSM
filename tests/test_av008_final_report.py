"""AV-008 report verification tests."""

import csv
from pathlib import Path

from scripts.fusion.final_report import verify


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

