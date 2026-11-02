"""Registered binary metrics for each disease task."""

from __future__ import annotations

from typing import Any

import torch

from chimera_ml.core.registry import METRICS


def _counts(predictions: torch.Tensor, targets: torch.Tensor,
            observed: torch.Tensor, task: int) -> tuple[int, int, int, int, int]:
    mask = observed[:, task].bool()
    predictions = predictions[mask, task].int()
    targets = targets[mask, task].int()
    tp = int(((predictions == 1) & (targets == 1)).sum())
    tn = int(((predictions == 0) & (targets == 0)).sum())
    fp = int(((predictions == 1) & (targets == 0)).sum())
    fn = int(((predictions == 0) & (targets == 1)).sum())
    return int(mask.sum()), tp, tn, fp, fn


def _metric(predictions: torch.Tensor, targets: torch.Tensor,
            observed: torch.Tensor, task: int, name: str) -> float:
    count, tp, tn, fp, fn = _counts(predictions, targets, observed, task)
    patient_recall = tp / max(tp + fn, 1)
    healthy_recall = tn / max(tn + fp, 1)
    patient_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    healthy_f1 = 2 * tn / max(2 * tn + fp + fn, 1)
    values = {
        "Macro F1": (patient_f1 + healthy_f1) / 2,
        "Weighted F1": (patient_f1 * (tp + fn) + healthy_f1 * (tn + fp)) / max(count, 1),
        "Macro Recall": (patient_recall + healthy_recall) / 2,
        "Weighted Recall": (patient_recall * (tp + fn) + healthy_recall * (tn + fp)) / max(count, 1),
        "Patient Recall": patient_recall,
        "Healthy Recall": healthy_recall,
    }
    return values[name]


@METRICS.register("carewsm_macro_f1")
def macro_f1(predictions, targets, observed, task=0, **_: Any) -> float:
    return _metric(predictions, targets, observed, task, "Macro F1")


@METRICS.register("carewsm_weighted_f1")
def weighted_f1(predictions, targets, observed, task=0, **_: Any) -> float:
    return _metric(predictions, targets, observed, task, "Weighted F1")


@METRICS.register("carewsm_macro_recall")
def macro_recall(predictions, targets, observed, task=0, **_: Any) -> float:
    return _metric(predictions, targets, observed, task, "Macro Recall")


@METRICS.register("carewsm_weighted_recall")
def weighted_recall(predictions, targets, observed, task=0, **_: Any) -> float:
    return _metric(predictions, targets, observed, task, "Weighted Recall")


@METRICS.register("carewsm_patient_recall")
def patient_recall(predictions, targets, observed, task=0, **_: Any) -> float:
    return _metric(predictions, targets, observed, task, "Patient Recall")


@METRICS.register("carewsm_healthy_recall")
def healthy_recall(predictions, targets, observed, task=0, **_: Any) -> float:
    return _metric(predictions, targets, observed, task, "Healthy Recall")


def corpus_average(care: dict[str, float], wsm: dict[str, float]) -> dict[str, float]:
    """Average separate CARE/WSM metric blocks without pooling predictions."""
    return {name: (care[name] + wsm[name]) / 2 for name in care}
