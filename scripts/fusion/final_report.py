#!/usr/bin/env python3
"""Verify collector outputs and build the AV-008 fusion research report."""

from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
from pathlib import Path
from typing import Iterable


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
GUIDED_METRICS = ("val_overall_uar", "test_overall_uar") + tuple(
    f"{split}_{component}_macro_recall"
    for split in ("val", "test")
    for component in COMPONENTS
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"invalid metric {field}: {value}")
    return value


def verify(rows: list[dict[str, str]], summaries: list[dict[str, str]]) -> dict[str, int]:
    """Audit run identity and each aggregate directly against the five fold rows."""
    if len(rows) != 285 or len(summaries) != 57:
        raise ValueError(
            f"expected 285 runs and 57 variants, got {len(rows)} and {len(summaries)}"
        )
    if any(row["status"] != "COMPLETED" for row in rows):
        raise ValueError("one or more runs are not COMPLETED")
    by_variant: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        variant, fold = row["experiment_id"], row["fold"]
        if fold not in {"0", "1", "2", "3", "4"}:
            raise ValueError(f"invalid fold: {variant}/{fold}")
        group = by_variant.setdefault(variant, {})
        if fold in group:
            raise ValueError(f"duplicate experiment/fold: {variant}/{fold}")
        group[fold] = row
        for split in ("val", "test"):
            for component in COMPONENTS:
                for metric in METRICS:
                    number(row, f"{split}_{component}_{metric}")
                macro = number(row, f"{split}_{component}_macro_recall")
                patient = number(row, f"{split}_{component}_patient_recall")
                healthy = number(row, f"{split}_{component}_healthy_recall")
                if abs(macro - (patient + healthy) / 2) > 0.001:
                    raise ValueError(
                        f"macro recall mismatch: {row['run_name']} {split}/{component}"
                    )
            expected = statistics.mean(
                number(row, f"{split}_{component}_macro_recall")
                for component in COMPONENTS
            )
            if abs(number(row, f"{split}_overall_uar") - expected) > 0.001:
                raise ValueError(f"overall UAR mismatch: {row['run_name']} {split}")
    if len(by_variant) != 57:
        raise ValueError(f"expected 57 unique variants, found {len(by_variant)}")
    summary_by_id: dict[str, dict[str, str]] = {}
    for summary in summaries:
        variant = summary["experiment_id"]
        if variant in summary_by_id:
            raise ValueError(f"duplicate five-fold summary: {variant}")
        summary_by_id[variant] = summary
    if set(summary_by_id) != set(by_variant):
        raise ValueError("summary identities differ from verified runs")
    metric_fields = [
        f"{split}_{component}_{metric}"
        for split in ("val", "test")
        for component in COMPONENTS
        for metric in METRICS
    ]
    fields = ["val_overall_uar", "test_overall_uar", *metric_fields]
    for variant, folds in by_variant.items():
        if set(folds) != {"0", "1", "2", "3", "4"}:
            raise ValueError(f"incomplete five-fold experiment: {variant}")
        summary = summary_by_id[variant]
        if (
            summary["five_fold_complete"] != "yes"
            or str(summary["completed_folds"]) != "5"
        ):
            raise ValueError(f"summary incorrectly marks five-fold complete: {variant}")
        first = folds["0"]
        for key in ("model", "architecture", "modalities", "feature_ids"):
            if summary[key] != first[key] or any(
                folds[fold][key] != first[key] for fold in folds
            ):
                raise ValueError(f"inconsistent {key} for {variant}")
        for field in fields:
            values = [number(folds[str(fold)], field) for fold in range(5)]
            expected_mean = statistics.mean(values)
            expected_sd = statistics.stdev(values)
            reported_mean = float(summary[f"{field}_mean"])
            reported_sd = float(summary[f"{field}_sd"])
            if (
                not math.isfinite(reported_mean)
                or not math.isfinite(reported_sd)
                or abs(reported_mean - expected_mean) > 2e-5
                or abs(reported_sd - expected_sd) > 2e-5
            ):
                raise ValueError(f"five-fold aggregate mismatch: {variant}/{field}")
    return {"runs": len(rows), "variants": len(summaries), "metrics_per_run": 48}


def verify_ranking_and_guided(
    rows: list[dict[str, str]],
    summaries: list[dict[str, str]],
    ranking: list[dict[str, str]],
    guided: list[dict[str, str]],
) -> None:
    """Check that VAL ranking and guided pairs match the audited fold data."""
    ordered = sorted(
        summaries,
        key=lambda r: (-float(r["val_overall_uar_mean"]), r["experiment_id"]),
    )
    if len(ranking) != 57:
        raise ValueError("expected 57 ranked variants")
    for position, (expected, actual) in enumerate(zip(ordered, ranking, strict=True), 1):
        if (
            actual["experiment_id"] != expected["experiment_id"]
            or actual["rank"] != str(position)
            or abs(
                float(actual["val_overall_uar_mean"])
                - float(expected["val_overall_uar_mean"])
            ) > 1e-8
        ):
            raise ValueError(f"ranking does not follow five-fold VAL: rank {position}")
    lookup = {(row["experiment_id"], row["fold"]): row for row in rows}
    text_variants = {
        row["experiment_id"].split("__", 1)[1]
        for row in summaries
        if row["architecture"] == "text_guided"
    }
    if len(text_variants) != 6 or len(guided) != 30:
        raise ValueError("expected six matched guided feature variants / 30 folds")
    seen: set[tuple[str, str]] = set()
    for pair in guided:
        key = (pair["feature_variant"], pair["fold"])
        if key in seen or key[0] not in text_variants or key[1] not in "01234":
            raise ValueError(f"duplicate or unexpected guided pair: {key}")
        seen.add(key)
        left = lookup.get((f"text_guided__{key[0]}", key[1]))
        right = lookup.get((f"audio_guided__{key[0]}", key[1]))
        if left is None or right is None:
            raise ValueError(f"missing matched guided fold: {key}")
        for metric in GUIDED_METRICS:
            reported = float(pair[f"{metric}_audio_minus_text"])
            expected = float(right[metric]) - float(left[metric])
            if not math.isfinite(reported) or abs(reported - expected) > 1e-8:
                raise ValueError(f"guided difference mismatch: {key}/{metric}")
    if len(seen) != 30:
        raise ValueError("incomplete guided comparisons")


def grouped_mean(rows: Iterable[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return statistics.mean(values)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def architecture_rows(summaries: list[dict[str, str]]) -> list[dict[str, object]]:
    result = []
    for architecture in sorted({row["architecture"] for row in summaries}):
        group = [row for row in summaries if row["architecture"] == architecture]
        values = [float(row["val_overall_uar_mean"]) for row in group]
        result.append({
            "architecture": architecture,
            "variants": len(group),
            "val_overall_uar_mean": statistics.mean(values),
            "val_overall_uar_sd_across_variants": statistics.stdev(values) if len(values) > 1 else 0.0,
            "best_variant": max(group, key=lambda row: float(row["val_overall_uar_mean"]))["experiment_id"],
        })
    return result


def best_rows(summaries: list[dict[str, str]]) -> list[dict[str, object]]:
    categories = {
        "overall": lambda row: True,
        "bimodal": lambda row: row["model"].startswith("carewsm_bimodal"),
        "trimodal": lambda row: not row["model"].startswith("carewsm_bimodal"),
        "audio_video": lambda row: row["modalities"] == "audio+video",
        "audio_text": lambda row: row["modalities"] == "audio+text",
        "video_text": lambda row: row["modalities"] == "video+text",
        "care_depression": lambda row: True,
        "care_parkinson": lambda row: True,
        "wsm_depression": lambda row: True,
        "wsm_parkinson": lambda row: True,
    }
    fields = {
        "overall": "val_overall_uar_mean",
        "bimodal": "val_overall_uar_mean",
        "trimodal": "val_overall_uar_mean",
        "audio_video": "val_overall_uar_mean",
        "audio_text": "val_overall_uar_mean",
        "video_text": "val_overall_uar_mean",
        "care_depression": "val_care_depression_macro_recall_mean",
        "care_parkinson": "val_care_parkinson_macro_recall_mean",
        "wsm_depression": "val_wsm_depression_macro_recall_mean",
        "wsm_parkinson": "val_wsm_parkinson_macro_recall_mean",
    }
    result = []
    for category, predicate in categories.items():
        eligible = [row for row in summaries if predicate(row)]
        if not eligible:
            continue
        field = fields[category]
        winner = max(eligible, key=lambda row: float(row[field]))
        descriptive_test_field = "test_" + field.removeprefix("val_")
        result.append({
            "category": category,
            "selection_metric": field,
            "experiment_id": winner["experiment_id"],
            "model": winner["model"],
            "feature_ids": winner["feature_ids"],
            "val_mean": winner[field],
            "val_sd": winner[field.replace("_mean", "_sd")],
            "test_selected_metric": descriptive_test_field,
            "test_selected_metric_mean": winner[descriptive_test_field],
            "test_selected_metric_sd": winner[descriptive_test_field.replace("_mean", "_sd")],
            "test_overall_uar_mean": winner["test_overall_uar_mean"],
            "test_overall_uar_sd": winner["test_overall_uar_sd"],
        })
    return result


def guided_summary(guided: list[dict[str, str]]) -> list[dict[str, object]]:
    result = []
    for metric in GUIDED_METRICS:
        values = [float(row[f"{metric}_audio_minus_text"]) for row in guided]
        result.append({
            "metric": metric,
            "mean_audio_minus_text": statistics.mean(values),
            "sd_audio_minus_text": statistics.stdev(values),
            "audio_wins": sum(value > 1e-12 for value in values),
            "text_wins": sum(value < -1e-12 for value in values),
            "ties": sum(abs(value) <= 1e-12 for value in values),
            "paired_folds": len(values),
        })
    return result


def feature_ablation(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    lookup = {
        (
            row["architecture"],
            row["modalities"],
            row["fold"],
            row["audio_feature"],
            row["video_feature"],
            row["text_feature"],
        ): row
        for row in rows
    }
    result = []
    for key, left in sorted(lookup.items()):
        architecture, modalities, fold, audio, video, text = key
        if audio:
            for other_audio in ("audio_id1", "audio_id2", "audio_id4"):
                if other_audio <= audio:
                    continue
                right = lookup.get((architecture, modalities, fold, other_audio, video, text))
                if right is not None:
                    result.append({
                        "comparison_type": "audio_version",
                        "architecture": architecture,
                        "modalities": modalities,
                        "fold": fold,
                        "left_features": "+".join(value for value in (audio, video, text) if value),
                        "right_features": "+".join(value for value in (other_audio, video, text) if value),
                        "val_overall_uar_right_minus_left": float(right["val_overall_uar"]) - float(left["val_overall_uar"]),
                        "test_overall_uar_right_minus_left": float(right["test_overall_uar"]) - float(left["test_overall_uar"]),
                    })
        if text:
            other_text = "text_id10" if text == "text_id9" else "text_id9"
            right = lookup.get((architecture, modalities, fold, audio, video, other_text))
            if right is not None and text == "text_id9":
                result.append({
                    "comparison_type": "text_version",
                    "architecture": architecture,
                    "modalities": modalities,
                    "fold": fold,
                    "left_features": "+".join(value for value in (audio, video, text) if value),
                    "right_features": "+".join(value for value in (audio, video, other_text) if value),
                    "val_overall_uar_right_minus_left": float(right["val_overall_uar"]) - float(left["val_overall_uar"]),
                    "test_overall_uar_right_minus_left": float(right["test_overall_uar"]) - float(left["test_overall_uar"]),
                })
    return result


def svg_bars(path: Path, labels: list[str], values: list[float], title: str) -> None:
    width, height = 1100, 560
    left, bottom, chart_w, chart_h = 270, 70, 790, 430
    maximum = max(values) if values else 1.0
    rows = []
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = bottom + chart_h - (index + 1) * chart_h / len(values)
        bar_w = max(0.0, value / maximum * chart_w)
        rows.append(
            f'<text x="{left - 8}" y="{y + 12:.1f}" text-anchor="end" font-size="12">{html.escape(label)}</text>'
            f'<rect x="{left}" y="{y:.1f}" width="{bar_w:.1f}" height="18" fill="#3465a4"/>'
            f'<text x="{left + bar_w + 5:.1f}" y="{y + 13:.1f}" font-size="11">{value:.3f}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="white"/><text x="550" y="28" text-anchor="middle" font-size="18">{html.escape(title)}</text>'
        f'<line x1="{left}" y1="{bottom}" x2="{left}" y2="{bottom + chart_h}" stroke="black"/>'
        f'<line x1="{left}" y1="{bottom + chart_h}" x2="{left + chart_w}" y2="{bottom + chart_h}" stroke="black"/>'
        + "".join(rows)
        + "</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def generate(root: Path) -> dict[str, int]:
    source = root / "logs/fusion/sweep/reports"
    report_dir = root / "reports/av_fusion_sweep"
    figure_dir = report_dir / "figures"
    rows = read_csv(source / "runs_status.csv")
    summaries = read_csv(source / "results_summary.csv")
    guided = read_csv(source / "comparison_audio_vs_text_guided.csv")
    ranking = read_csv(source / "ranking_val.csv")
    counts = verify(rows, summaries)
    verify_ranking_and_guided(rows, summaries, ranking, guided)
    architectures = architecture_rows(summaries)
    best = best_rows(summaries)
    guided_rows = guided_summary(guided)
    ablation_rows = feature_ablation(rows)
    write_csv(report_dir / "bimodal_comparison.csv", [row for row in architectures if row["architecture"].startswith("bimodal")])
    write_csv(report_dir / "trimodal_comparison.csv", [row for row in architectures if not row["architecture"].startswith("bimodal")])
    write_csv(report_dir / "best_by_task.csv", best)
    write_csv(report_dir / "guided_comparison_summary.csv", guided_rows)
    write_csv(report_dir / "feature_ablation.csv", ablation_rows)
    figure_dir.mkdir(parents=True, exist_ok=True)
    top = sorted(summaries, key=lambda row: float(row["val_overall_uar_mean"]), reverse=True)[:15]
    svg_bars(
        figure_dir / "val_overall_uar_top15.svg",
        [row["experiment_id"] for row in top],
        [float(row["val_overall_uar_mean"]) for row in top],
        "Top 15 configurations by five-fold VAL Overall UAR",
    )
    svg_bars(
        figure_dir / "architecture_val_means.svg",
        [str(row["architecture"]) for row in architectures],
        [float(row["val_overall_uar_mean"]) for row in architectures],
        "Architecture mean across compatible variants",
    )
    guided_overall = [row for row in guided_rows if row["metric"] in ("val_overall_uar", "test_overall_uar")]
    svg_bars(
        figure_dir / "audio_minus_text_overall.svg",
        [str(row["metric"]) for row in guided_overall],
        [float(row["mean_audio_minus_text"]) for row in guided_overall],
        "Audio-Guided minus Text-Guided mean paired difference",
    )
    base = (source / "FINAL_RESEARCH_REPORT.md").read_text(encoding="utf-8")
    detailed = base[base.index("## Detailed VAL metrics") :]
    overall = max(summaries, key=lambda row: float(row["val_overall_uar_mean"]))
    arch_table = "\n".join(
        f"| `{row['architecture']}` | {row['variants']} | {float(row['val_overall_uar_mean']):.4f} | `{row['best_variant']}` |"
        for row in architectures
    )
    best_table = "\n".join(
        f"| {row['category']} | {row['selection_metric']} | `{row['experiment_id']}` | `{row['feature_ids']}` | {float(row['val_mean']):.4f} ± {float(row['val_sd']):.4f} | `{row['test_selected_metric']}`: {float(row['test_selected_metric_mean']):.4f} ± {float(row['test_selected_metric_sd']):.4f} | {float(row['test_overall_uar_mean']):.4f} ± {float(row['test_overall_uar_sd']):.4f} |"
        for row in best
    )
    guided_table = "\n".join(
        f"| `{row['metric']}` | {float(row['mean_audio_minus_text']):+.4f} ± {float(row['sd_audio_minus_text']):.4f} | {row['audio_wins']} | {row['text_wins']} | {row['ties']} |"
        for row in guided_rows
    )
    prefix = f'''# AV-008 Final Fusion Research Report

`UPSTREAM_PROVENANCE=UNKNOWN — EXPLORATORY RESULTS ONLY`

## 1. Executive summary and key findings

The verified sweep contains **{counts['runs']}/285 runs**, **{counts['variants']}/57 five-fold variants**, and **{counts['metrics_per_run']} metrics per run** (24 VAL + 24 TEST). All TEST results are descriptive and were evaluated from the original selected VAL checkpoints; no TEST value was used for selection.

The best prespecified configuration is `{overall['experiment_id']}` (`{overall['feature_ids']}`), with VAL Overall UAR **{float(overall['val_overall_uar_mean']):.4f} ± {float(overall['val_overall_uar_sd']):.4f}**. Its descriptive TEST Overall UAR is **{float(overall['test_overall_uar_mean']):.4f} ± {float(overall['test_overall_uar_sd']):.4f}**.

## 2. Experiment design and data provenance

The matrix contains 165 bimodal jobs (3 architectures × 11 compatible feature selections × 5 folds) and 120 trimodal jobs (4 architectures × 6 selections × 5 folds). Selection is the five-fold mean VAL Overall UAR. Fold aggregation uses arithmetic mean and sample SD (`ddof=1`).

## 3. Completion and verification

All runtime identities, selected checkpoints, VAL monitors, config hashes, recovery provenance records and 48 metric fields passed verification. No participant-level records or source paths are included in this report. The local collector artifacts remain under `logs/` and are not committed.

## 4. Full VAL results

The complete six-metric VAL tables follow in the collector-derived section below.

## 5. Full descriptive TEST results

The complete six-metric TEST tables follow below. They are not a selection signal.

## 6. Bimodal and trimodal architecture comparisons

| Architecture | Variants | Mean VAL Overall UAR | Best variant |
|---|---:|---:|---|
{arch_table}

These architecture means average compatible feature variants and are descriptive; feature coverage differs by modality family. The complete split-by-task tables are retained below.

## 7. Audio-Guided vs Text-Guided

The matched table contains 30 paired folds (six feature combinations × five folds). Differences are Audio-Guided minus Text-Guided; no significance claim is made.

| Metric | Mean difference ± sample SD | Audio wins | Text wins | Ties |
|---|---:|---:|---:|---:|
{guided_table}

## 8. Audio/text feature-version and modality ablations

`bimodal_comparison.csv`, `trimodal_comparison.csv`, `feature_ablation.csv`, and `guided_comparison_summary.csv` are matched-design summaries. Audio-version comparisons are interpreted only within the same architecture, modality set, text version and fold; text-version comparisons are interpreted only within the same architecture, audio version and fold. Overall modality-family means must not be read as causal feature effects because architecture and input set also change.

## 9. Best models by overall and individual tasks

Task-specific rows use VAL Macro Recall as the exploratory task metric; they are not replacements for the prespecified overall ranking.

| Category | VAL selection metric | Experiment | Features | VAL mean ± SD | Matched descriptive TEST metric mean ± SD | TEST Overall UAR mean ± SD |
|---|---|---|---|---:|---:|---:|
{best_table}

## 10. Interpretation, limitations and next research questions

These are exploratory results. Upstream FIT/SELECTION identities remain unknown. TEST measurements were not used to choose features, architectures, checkpoints or winners. No leakage-free generalization, clinical validity or statistical significance claim is made. Follow-up work should pre-register any confirmatory comparison and preserve the validation-only decision rule.

## 11. Reproducibility

The report is generated from `logs/fusion/sweep/reports/*.csv` by `scripts/fusion/final_report.py`; the collector is `scripts/fusion/collect_results.py`. Figures are deterministic SVG files under `reports/av_fusion_sweep/figures/`. Training logs, checkpoints, raw features and MLflow databases are excluded from Git.

## 12. Figures

- `figures/val_overall_uar_top15.svg`
- `figures/architecture_val_means.svg`
- `figures/audio_minus_text_overall.svg`

{detailed}
'''
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "FINAL_RESEARCH_REPORT.md").write_text(prefix, encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    counts = generate(args.root.resolve())
    print("verified " + ", ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
