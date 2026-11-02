"""Stateful binary metrics for the Chimera-ML 0.3.0 interface."""

from __future__ import annotations

from typing import Any

import torch

from chimera_ml.core.registry import METRICS
from chimera_ml.metrics.base import BaseMetric


def _metric_values(predictions: torch.Tensor, targets: torch.Tensor, observed: torch.Tensor, task: int) -> dict[str, float]:
    mask = observed[:, task].bool()
    prediction = predictions[mask, task].int()
    target = targets[mask, task].int()
    true_positive = int(((prediction == 1) & (target == 1)).sum())
    true_negative = int(((prediction == 0) & (target == 0)).sum())
    false_positive = int(((prediction == 1) & (target == 0)).sum())
    false_negative = int(((prediction == 0) & (target == 1)).sum())
    count = int(mask.sum())
    patient_recall = true_positive / max(true_positive + false_negative, 1)
    healthy_recall = true_negative / max(true_negative + false_positive, 1)
    patient_f1 = 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)
    healthy_f1 = 2 * true_negative / max(2 * true_negative + false_positive + false_negative, 1)
    return {
        "count": float(count),
        "Macro F1": (patient_f1 + healthy_f1) / 2,
        "Weighted F1": (patient_f1 * (true_positive + false_negative) + healthy_f1 * (true_negative + false_positive)) / max(count, 1),
        "Macro Recall": (patient_recall + healthy_recall) / 2,
        "Weighted Recall": (patient_recall * (true_positive + false_negative) + healthy_recall * (true_negative + false_positive)) / max(count, 1),
        "Patient Recall": patient_recall,
        "Healthy Recall": healthy_recall,
    }


class BinaryMetric(BaseMetric):
    """Accumulate one required metric for one disease task."""

    def __init__(self, name: str, task: int = 0) -> None:
        self.name = name
        self.task = int(task)
        self.reset()

    def reset(self) -> None:
        self.predictions: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self.observed: list[torch.Tensor] = []

    def update(self, output: Any, batch: Any) -> None:
        observed = batch.get_masks("observed_targets")
        if observed is None and batch.meta and "sample_meta" in batch.meta:
            observed = torch.stack([item["observed_targets"] for item in batch.meta["sample_meta"]])
        if observed is None:
            raise ValueError("Batch does not contain observed-label masks")
        self.predictions.append((output.preds > 0).detach().cpu())
        self.targets.append(batch.targets.detach().cpu())
        self.observed.append(observed.detach().cpu())

    def compute(self) -> dict[str, float]:
        if not self.predictions:
            return {self.name: 0.0, "count": 0.0}
        values = _metric_values(
            torch.cat(self.predictions), torch.cat(self.targets), torch.cat(self.observed), self.task
        )
        return {self.name: values[self.name], "count": values["count"]}


@METRICS.register("carewsm_macro_f1")
def carewsm_macro_f1(context: Any | None = None, **params: Any) -> BinaryMetric:
    del context
    return BinaryMetric("Macro F1", **params)


@METRICS.register("carewsm_weighted_f1")
def carewsm_weighted_f1(context: Any | None = None, **params: Any) -> BinaryMetric:
    del context
    return BinaryMetric("Weighted F1", **params)


@METRICS.register("carewsm_macro_recall")
def carewsm_macro_recall(context: Any | None = None, **params: Any) -> BinaryMetric:
    del context
    return BinaryMetric("Macro Recall", **params)


@METRICS.register("carewsm_weighted_recall")
def carewsm_weighted_recall(context: Any | None = None, **params: Any) -> BinaryMetric:
    del context
    return BinaryMetric("Weighted Recall", **params)


@METRICS.register("carewsm_patient_recall")
def carewsm_patient_recall(context: Any | None = None, **params: Any) -> BinaryMetric:
    del context
    return BinaryMetric("Patient Recall", **params)


@METRICS.register("carewsm_healthy_recall")
def carewsm_healthy_recall(context: Any | None = None, **params: Any) -> BinaryMetric:
    del context
    return BinaryMetric("Healthy Recall", **params)


def corpus_average(care: dict[str, float], wsm: dict[str, float]) -> dict[str, float]:
    return {name: (care[name] + wsm[name]) / 2 for name in care}
