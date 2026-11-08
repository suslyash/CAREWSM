"""Corpus-specific combined metric monitor."""

from __future__ import annotations

import math
from typing import Any

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


_METRICS = (
    "macro_f1",
    "weighted_f1",
    "macro_recall",
    "weighted_recall",
    "patient_recall",
    "healthy_recall",
)
_TASKS = ("depression", "parkinson")
_CORPORA = ("care", "wsm")


class CAREWSMValidationMonitorCallback(BaseCallback):
    """Add equal-weight corpus/task aggregates to epoch logs and MLflow."""

    components = (
        "val_care/depression/macro_recall",
        "val_care/parkinson/macro_recall",
        "val_wsm/depression/macro_recall",
        "val_wsm/parkinson/macro_recall",
    )

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        validation_values = [logs.get(name) for name in self.components]
        invalid = [
            name
            for name, value in zip(self.components, validation_values)
            if value is None or not math.isfinite(float(value))
        ]
        if invalid:
            raise ValueError(
                "val_mean_corpora_tasks/macro_recall is unavailable; required "
                f"validation components are missing or invalid: {', '.join(invalid)}"
            )

        derived: dict[str, float] = {}
        for split in ("val", "test"):
            for task in _TASKS:
                for metric in _METRICS:
                    values = [
                        logs.get(f"{split}_{corpus}/{task}/{metric}")
                        for corpus in _CORPORA
                    ]
                    key = f"{split}_mean_corpora/{task}/{metric}"
                    derived[key] = self._mean(values)

        for metric in _METRICS:
            validation_values = [
                logs.get(f"{corpus}/{task}/{metric}")
                for corpus in ("val_care", "val_wsm")
                for task in _TASKS
            ]
            validation_key = f"val_mean_corpora_tasks/{metric}"
            derived[validation_key] = (
                self._required_mean(validation_values, validation_key)
                if metric == "macro_recall"
                else self._mean(validation_values)
            )
            test_values = [
                logs.get(f"test_{corpus}/{task}/{metric}")
                for corpus in _CORPORA
                for task in _TASKS
            ]
            derived[f"test_mean_corpora_tasks/{metric}"] = self._mean(test_values)

        logs.update(derived)
        logs["dev_mean_macro_recall"] = derived[
            "val_mean_corpora_tasks/macro_recall"
        ]

        mlflow_logger = getattr(trainer, "mlflow_logger", None)
        if mlflow_logger is not None:
            mlflow_logger.log_metrics(
                {
                    key: value
                    for key, value in derived.items()
                    if math.isfinite(value)
                },
                step=epoch,
            )

        values = []
        for key, value in derived.items():
            formatted = (
                f"{key}={value:.6f}"
                if math.isfinite(value)
                else f"{key}=nan"
            )
            values.append(formatted)
        self._info(trainer, f"[carewsm_combined epoch {epoch}] " + " | ".join(values))

    @staticmethod
    def _mean(values: list[float | None]) -> float:
        if any(value is None or not math.isfinite(float(value)) for value in values):
            return float("nan")
        return sum(float(value) for value in values) / len(values)

    @classmethod
    def _required_mean(cls, values: list[float | None], metric: str) -> float:
        result = cls._mean(values)
        if not math.isfinite(result):
            raise ValueError(
                f"{metric} is unavailable; required validation "
                "components are missing or invalid"
            )
        return result


@CALLBACKS.register("carewsm_validation_monitor")
def carewsm_validation_monitor(
    context: Any | None = None,
    **params: Any,
) -> CAREWSMValidationMonitorCallback:
    del context, params
    return CAREWSMValidationMonitorCallback()


class CAREWSMTaskValidationMonitorCallback(BaseCallback):
    """Aggregate only the active task for single-task checkpoint selection."""

    def __init__(self, task: str) -> None:
        if task not in _TASKS:
            raise ValueError("task must be depression or parkinson")
        self.task = task

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        derived: dict[str, float] = {}
        for split in ("val", "test"):
            for metric in _METRICS:
                values = [
                    logs.get(f"{split}_{corpus}/{self.task}/{metric}")
                    for corpus in _CORPORA
                ]
                key = f"{split}_mean_corpora/{self.task}/{metric}"
                derived[key] = CAREWSMValidationMonitorCallback._mean(values)
        monitor = f"val_mean_corpora/{self.task}/macro_recall"
        if not math.isfinite(derived[monitor]):
            raise ValueError(
                f"{monitor} is unavailable; both corpus validation components "
                "must be finite"
            )
        logs.update(derived)
        mlflow_logger = getattr(trainer, "mlflow_logger", None)
        if mlflow_logger is not None:
            mlflow_logger.log_metrics(
                {key: value for key, value in derived.items() if math.isfinite(value)},
                step=epoch,
            )
        self._info(
            trainer,
            f"[carewsm_{self.task} epoch {epoch}] "
            f"{monitor}={derived[monitor]:.6f}",
        )


@CALLBACKS.register("carewsm_task_validation_monitor")
def carewsm_task_validation_monitor(
    context: Any | None = None, **params: Any
) -> CAREWSMTaskValidationMonitorCallback:
    del context
    return CAREWSMTaskValidationMonitorCallback(**params)
