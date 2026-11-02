"""Stateful corpus-aware binary metrics for CAREWSM."""

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


def _metric_values(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    observed: torch.Tensor,
    task: int,
) -> dict[str, float]:
    mask = observed[:, task].bool()
    count = int(mask.sum())
    if count == 0:
        return {name: float("nan") for name in _METRICS} | {"observed_count": 0.0}

    prediction = predictions[mask, task].int()
    target = targets[mask, task].int()
    true_positive = int(((prediction == 1) & (target == 1)).sum())
    true_negative = int(((prediction == 0) & (target == 0)).sum())
    false_positive = int(((prediction == 1) & (target == 0)).sum())
    false_negative = int(((prediction == 0) & (target == 1)).sum())

    patient_support = true_positive + false_negative
    healthy_support = true_negative + false_positive
    patient_recall = true_positive / max(patient_support, 1)
    healthy_recall = true_negative / max(healthy_support, 1)
    patient_f1 = 2 * true_positive / max(
        2 * true_positive + false_positive + false_negative, 1
    )
    healthy_f1 = 2 * true_negative / max(
        2 * true_negative + false_positive + false_negative, 1
    )

    return {
        "observed_count": float(count),
        "macro_f1": (patient_f1 + healthy_f1) / 2,
        "weighted_f1": (
            patient_f1 * patient_support + healthy_f1 * healthy_support
        ) / max(count, 1),
        "macro_recall": (patient_recall + healthy_recall) / 2,
        "weighted_recall": (
            patient_recall * patient_support + healthy_recall * healthy_support
        ) / max(count, 1),
        "patient_recall": patient_recall,
        "healthy_recall": healthy_recall,
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
            observed = torch.stack(
                [item["observed_targets"] for item in batch.meta["sample_meta"]]
            )
        if observed is None:
            raise ValueError("Batch does not contain observed-label masks")

        self.predictions.append((output.preds > 0).detach().cpu())
        self.targets.append(batch.targets.detach().cpu())
        self.observed.append(observed.detach().cpu())

    def compute(self) -> dict[str, float]:
        if not self.predictions:
            return {self.name: float("nan"), "observed_count": 0.0}

        values = _metric_values(
            torch.cat(self.predictions),
            torch.cat(self.targets),
            torch.cat(self.observed),
            self.task,
        )
        return {self.name: values[self.name], "observed_count": values["observed_count"]}


@METRICS.register("carewsm_macro_f1")
def carewsm_macro_f1(context: Any | None = None, **params: Any) -> BinaryMetric:
    return BinaryMetric("macro_f1", **params)


@METRICS.register("carewsm_weighted_f1")
def carewsm_weighted_f1(context: Any | None = None, **params: Any) -> BinaryMetric:
    return BinaryMetric("weighted_f1", **params)


@METRICS.register("carewsm_macro_recall")
def carewsm_macro_recall(context: Any | None = None, **params: Any) -> BinaryMetric:
    return BinaryMetric("macro_recall", **params)


@METRICS.register("carewsm_weighted_recall")
def carewsm_weighted_recall(context: Any | None = None, **params: Any) -> BinaryMetric:
    return BinaryMetric("weighted_recall", **params)


@METRICS.register("carewsm_patient_recall")
def carewsm_patient_recall(context: Any | None = None, **params: Any) -> BinaryMetric:
    return BinaryMetric("patient_recall", **params)


@METRICS.register("carewsm_healthy_recall")
def carewsm_healthy_recall(context: Any | None = None, **params: Any) -> BinaryMetric:
    return BinaryMetric("healthy_recall", **params)


def corpus_average(
    care: dict[str, float],
    wsm: dict[str, float],
) -> dict[str, float]:
    return {name: (care[name] + wsm[name]) / 2 for name in care}


class MultitaskPRFMetric(BaseMetric):
    """Six metrics for both diseases with separate corpus averages."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._rows: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]
        ] = []

    def update(self, output: Any, batch: Any) -> None:
        observed = batch.get_masks("observed_targets")
        if observed is None and batch.meta and "sample_meta" in batch.meta:
            observed = torch.stack(
                [item["observed_targets"] for item in batch.meta["sample_meta"]]
            )
        if observed is None:
            raise ValueError("Batch does not contain observed-label masks")

        metas = batch.meta.get("sample_meta", []) if batch.meta else []
        self._rows.append(
            (
                output.preds.detach().cpu(),
                batch.targets.detach().cpu(),
                observed.detach().cpu(),
                metas,
            )
        )

    def compute(self) -> dict[str, float]:
        if not self._rows:
            return self._undefined("joint")

        predictions = torch.cat([row[0] for row in self._rows])
        targets = torch.cat([row[1] for row in self._rows])
        observed = torch.cat([row[2] for row in self._rows])
        metas = [meta for row in self._rows for meta in row[3]]
        corpora = [meta.get("corpus", "UNKNOWN") for meta in metas]

        result: dict[str, float] = {}
        corpus_results: dict[str, dict[str, float]] = {}
        for corpus in ("CARE", "WSM"):
            mask = torch.tensor(
                [item == corpus for item in corpora],
                dtype=torch.bool,
            )
            if mask.any():
                block = self._compute_block(
                    predictions[mask],
                    targets[mask],
                    observed[mask],
                    corpus.lower(),
                )
                result.update(block)
                corpus_results[corpus.lower()] = block

        if len(corpus_results) == 2:
            result.update(self._compute_block(predictions, targets, observed, "joint"))

        for task, _ in _TASKS:
            for name in _METRICS:
                care_value = corpus_results.get("care", {}).get(
                    f"care/{task}/{name}",
                    float("nan"),
                )
                wsm_value = corpus_results.get("wsm", {}).get(
                    f"wsm/{task}/{name}",
                    float("nan"),
                )
                result[f"average/{task}/{name}"] = (
                    (care_value + wsm_value) / 2
                    if care_value == care_value and wsm_value == wsm_value
                    else float("nan")
                )
            care_count = corpus_results.get("care", {}).get(
                f"care/{task}/observed_count",
                0.0,
            )
            wsm_count = corpus_results.get("wsm", {}).get(
                f"wsm/{task}/observed_count",
                0.0,
            )
            result[f"average/{task}/observed_count"] = (
                (care_count + wsm_count) / 2
                if care_count and wsm_count
                else 0.0
            )

        valid_task_scores = [
            result[f"average/{task}/macro_f1"]
            for task, _ in _TASKS
            if result[f"average/{task}/macro_f1"]
            == result[f"average/{task}/macro_f1"]
        ]
        result["average_macro_f1"] = (
            sum(valid_task_scores) / len(valid_task_scores)
            if valid_task_scores
            else float("-inf")
        )
        return result

    def _compute_block(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        observed: torch.Tensor,
        prefix: str,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for task, index in _TASKS:
            values = _metric_values(predictions > 0, targets, observed, index)
            for name in _METRICS:
                result[f"{prefix}/{task}/{name}"] = values[name]
            result[f"{prefix}/{task}/observed_count"] = values["observed_count"]
        return result

    def _undefined(self, prefix: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for task, _ in _TASKS:
            for name in _METRICS:
                result[f"{prefix}/{task}/{name}"] = float("nan")
            result[f"{prefix}/{task}/observed_count"] = 0.0
        result["average_macro_f1"] = float("nan")
        return result


@METRICS.register("carewsm_multitask_prf")
def carewsm_multitask_prf(
    context: Any | None = None,
    **params: Any,
) -> MultitaskPRFMetric:
    return MultitaskPRFMetric()
