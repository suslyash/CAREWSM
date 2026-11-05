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
    """Add equal-weight CARE/WSM validation and TEST averages to epoch logs."""

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
                "dev_mean_macro_recall is unavailable; required validation "
                f"components are missing or invalid: {', '.join(invalid)}"
            )

        for split, prefix in (("val", "val_average"), ("test", "test_average")):
            for task in _TASKS:
                for metric in _METRICS:
                    values = [
                        logs.get(f"{split}_{corpus}/{task}/{metric}")
                        for corpus in _CORPORA
                    ]
                    logs[f"{prefix}/{task}/{metric}"] = self._mean(values)

        for metric in _METRICS:
            values = [
                logs.get(f"{corpus}/{task}/{metric}")
                for corpus in ("val_care", "val_wsm")
                for task in _TASKS
            ]
            logs[f"dev_mean_{metric}"] = (
                self._required_mean(values, metric)
                if metric == "macro_recall"
                else self._mean(values)
            )
            test_values = [
                logs.get(f"test_{corpus}/{task}/{metric}")
                for corpus in _CORPORA
                for task in _TASKS
            ]
            logs[f"test_mean_{metric}"] = self._mean(test_values)

        values = []
        for key, value in logs.items():
            if key.startswith(
                ("val_average/", "test_average/", "dev_mean_", "test_mean_")
            ):
                formatted = f"{key}={value:.6f}" if math.isfinite(value) else f"{key}=nan"
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
                f"dev_mean_{metric} is unavailable; required validation "
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
