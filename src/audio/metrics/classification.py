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


class MultitaskPRFMetric(BaseMetric):
    """Six required metrics for both diseases, with corpus-aware keys."""

    _tasks = (("depression", 0), ("parkinson", 1))
    _metric_names = ("macro_f1", "weighted_f1", "macro_recall", "weighted_recall",
                     "patient_recall", "healthy_recall")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]] = []

    def update(self, output: Any, batch: Any) -> None:
        observed = batch.get_masks("observed_targets")
        if observed is None and batch.meta and "sample_meta" in batch.meta:
            observed = torch.stack([item["observed_targets"] for item in batch.meta["sample_meta"]])
        if observed is None:
            raise ValueError("Batch does not contain observed-label masks")
        metas = batch.meta.get("sample_meta", []) if batch.meta else []
        self._rows.append((output.preds.detach().cpu(), batch.targets.detach().cpu(),
                           observed.detach().cpu(), metas))

    def compute(self) -> dict[str, float]:
        if not self._rows:
            return self._undefined("joint")
        predictions = torch.cat([row[0] for row in self._rows])
        targets = torch.cat([row[1] for row in self._rows])
        observed = torch.cat([row[2] for row in self._rows])
        metas = [meta for row in self._rows for meta in row[3]]
        corpora = [meta.get("corpus", "UNKNOWN") for meta in metas]
        result: dict[str, float] = {}
        for corpus in ("CARE", "WSM"):
            mask = torch.tensor([item == corpus for item in corpora], dtype=torch.bool)
            if mask.any():
                result.update(self._compute_block(predictions[mask], targets[mask], observed[mask], corpus.lower()))
        result.update(self._compute_block(predictions, targets, observed, "joint"))
        if all(f"{corpus.lower()}/depression/{name}" in result for corpus in ("CARE", "WSM") for name in self._metric_names):
            for task, _ in self._tasks:
                for name in self._metric_names:
                    result[f"average/{task}/{name}"] = (result[f"care/{task}/{name}"] + result[f"wsm/{task}/{name}"]) / 2
        return result

    def _compute_block(self, predictions: torch.Tensor, targets: torch.Tensor,
                       observed: torch.Tensor, prefix: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for task, index in self._tasks:
            values = _metric_values(predictions > 0, targets, observed, index)
            count = values.pop("count")
            result[f"{prefix}/{task}/observed_count"] = count
            for name in self._metric_names:
                metric_name = name.replace(" ", "_").lower()
                metric_value = (values[{"macro_f1": "Macro F1", "weighted_f1": "Weighted F1", "macro_recall": "Macro Recall", "weighted_recall": "Weighted Recall", "patient_recall": "Patient Recall", "healthy_recall": "Healthy Recall"}[name]] if count else float("nan"))
                result[f"{prefix}/{task}/{metric_name}"] = metric_value
                if prefix == "joint":
                    result[f"{task}_{metric_name}"] = metric_value
        return result

    def _undefined(self, prefix: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for task, _ in self._tasks:
            result[f"{prefix}/{task}/observed_count"] = 0.0
            for name in self._metric_names:
                result[f"{prefix}/{task}/{name}"] = float("nan")
        return result


@METRICS.register("carewsm_multitask_prf")
def carewsm_multitask_prf(context: Any | None = None, **params: Any) -> MultitaskPRFMetric:
    del context, params
    return MultitaskPRFMetric()
