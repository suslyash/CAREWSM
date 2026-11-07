"""Validation-only controllers and diagnostics for the Stage G MTL suite."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


@dataclass
class CAREWSMStageGMTLCallback(BaseCallback):
    """Bind gradient losses and update Progress/RA-STCH from validation only."""

    def _loss(self, trainer: Any) -> Any:
        loss = getattr(trainer, "loss_fn", None)
        if loss is None or not hasattr(loss, "method"):
            raise TypeError("carewsm_stageg_mtl_callback requires the Stage G loss")
        return loss

    def on_fit_start(self, trainer: Any) -> None:
        loss = self._loss(trainer)
        model = getattr(trainer, "model", None)
        if model is None or not hasattr(loss, "bind_model"):
            raise TypeError("Stage G loss requires trainer.model")
        loss.bind_model(model)
        if loss.ownership_counts != {"shared": 12, "depression": 6, "parkinson": 6}:
            raise ValueError(f"unexpected gated-fusion ownership: {loss.ownership_counts}")

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        del epoch
        self._loss(trainer).reset_epoch_diagnostics()

    @staticmethod
    def _mean(logs: dict[str, float], task: str) -> float:
        names = (
            f"val_care/{task}/macro_recall",
            f"val_wsm/{task}/macro_recall",
        )
        values = [logs.get(name) for name in names]
        if any(value is None or not math.isfinite(float(value)) for value in values):
            raise KeyError(
                "two-corpus validation Macro Recall is unavailable: "
                + ", ".join(names)
            )
        return sum(float(value) for value in values) / len(values)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        loss = self._loss(trainer)
        logs.update(loss.epoch_diagnostics())
        if loss.method in {"progress", "ra_stch"}:
            depression = self._mean(logs, "depression")
            parkinson = self._mean(logs, "parkinson")
            loss.update_controller(depression, parkinson)
            logs["mtl/controller_weight_depression"] = float(loss.weights[0])
            logs["mtl/controller_weight_parkinson"] = float(loss.weights[1])
            logs["mtl/validation_feedback_depression"] = depression
            logs["mtl/validation_feedback_parkinson"] = parkinson
        logger = getattr(trainer, "mlflow_logger", None)
        if logger is not None:
            logger.log_metrics(
                {
                    key: value
                    for key, value in logs.items()
                    if key.startswith("mtl/") and math.isfinite(float(value))
                },
                step=epoch,
            )


@CALLBACKS.register("carewsm_stageg_mtl_callback")
def carewsm_stageg_mtl_callback(
    context: Any | None = None, **params: Any
) -> CAREWSMStageGMTLCallback:
    del context, params
    return CAREWSMStageGMTLCallback()
