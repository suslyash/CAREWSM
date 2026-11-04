"""Masked six-metric classification metric for CAREWSM."""

from __future__ import annotations

from typing import Any

import torch

from chimera_ml.core.registry import METRICS
from chimera_ml.metrics.base import BaseMetric

_TASKS = (("depression", 0), ("parkinson", 1))
_METRICS = (
    "macro_f1",
    "weighted_f1",
    "macro_recall",
    "weighted_recall",
    "patient_recall",
    "healthy_recall",
)

class CAREWSMClassificationMetric(BaseMetric):
    """Compute masked six-metric classification results for one loader."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._predictions: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []
        self._observed: list[torch.Tensor] = []

    def update(self, output: Any, batch: Any) -> None:
        observed = batch.get_masks("observed_targets")
        if observed is None and batch.meta and "sample_meta" in batch.meta:
            observed = torch.stack(
                [item["observed_targets"] for item in batch.meta["sample_meta"]]
            )
        if observed is None:
            raise ValueError("Batch does not contain observed-label masks")
        self._predictions.append((output.preds > 0).detach().cpu())
        self._targets.append(batch.targets.detach().cpu())
        self._observed.append(observed.detach().cpu().bool())

    def compute(self) -> dict[str, float]:
        if not self._predictions:
            return self._empty_result()

        predictions = torch.cat(self._predictions)
        targets = torch.cat(self._targets)
        observed = torch.cat(self._observed)
        result: dict[str, float] = {}
        for task, index in _TASKS:
            result.update(
                {
                    f"{task}/{name}": value
                    for name, value in self._task_values(
                        predictions[:, index], targets[:, index], observed[:, index]
                    ).items()
                }
            )
        return result

    @staticmethod
    def _task_values(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        observed: torch.Tensor,
    ) -> dict[str, float]:
        mask = observed.bool()
        count = int(mask.sum())
        if count == 0:
            return {
                **{name: float("nan") for name in _METRICS},
                "observed_count": 0.0,
                "positive_support": 0.0,
                "negative_support": 0.0,
            }

        prediction = predictions[mask].int()
        target = targets[mask].int()
        positive_support = int((target == 1).sum())
        negative_support = int((target == 0).sum())
        if positive_support == 0 or negative_support == 0:
            return {
                **{name: float("nan") for name in _METRICS},
                "observed_count": float(count),
                "positive_support": float(positive_support),
                "negative_support": float(negative_support),
            }

        true_positive = int(((prediction == 1) & (target == 1)).sum())
        true_negative = int(((prediction == 0) & (target == 0)).sum())
        false_positive = int(((prediction == 1) & (target == 0)).sum())
        false_negative = int(((prediction == 0) & (target == 1)).sum())
        patient_recall = true_positive / positive_support
        healthy_recall = true_negative / negative_support
        patient_f1 = 2 * true_positive / max(
            2 * true_positive + false_positive + false_negative, 1
        )
        healthy_f1 = 2 * true_negative / max(
            2 * true_negative + false_positive + false_negative, 1
        )
        return {
            "observed_count": float(count),
            "positive_support": float(positive_support),
            "negative_support": float(negative_support),
            "macro_f1": (patient_f1 + healthy_f1) / 2,
            "weighted_f1": (
                patient_f1 * positive_support + healthy_f1 * negative_support
            )
            / count,
            "macro_recall": (patient_recall + healthy_recall) / 2,
            "weighted_recall": (
                patient_recall * positive_support
                + healthy_recall * negative_support
            )
            / count,
            "patient_recall": patient_recall,
            "healthy_recall": healthy_recall,
        }

    @staticmethod
    def _empty_result() -> dict[str, float]:
        result: dict[str, float] = {}
        for task, _ in _TASKS:
            result.update({f"{task}/{name}": float("nan") for name in _METRICS})
            result[f"{task}/observed_count"] = 0.0
            result[f"{task}/positive_support"] = 0.0
            result[f"{task}/negative_support"] = 0.0
        return result


@METRICS.register("carewsm_masked_multitask_classification")
def carewsm_masked_multitask_classification(
    context: Any | None = None,
    **params: Any,
) -> CAREWSMClassificationMetric:
    del context, params
    return CAREWSMClassificationMetric()
