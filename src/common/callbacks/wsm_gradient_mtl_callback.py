"""Epoch logging for TASK-010A parameter-gradient comparator diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


@dataclass
class WSMGradientMTLCallback(BaseCallback):
    """Expose train-only gradient-combiner diagnostics at each DEV epoch."""

    def _loss(self, trainer: Any) -> Any:
        loss = getattr(trainer, "loss_fn", None)
        if loss is None or not hasattr(loss, "epoch_diagnostics") or not hasattr(loss, "reset_epoch_diagnostics"):
            raise TypeError("wsm_gradient_mtl_callback requires wsm_gradient_mtl_loss")
        return loss

    def on_fit_start(self, trainer: Any) -> None:
        loss = self._loss(trainer)
        model = getattr(trainer, "model", None)
        if model is None or not hasattr(loss, "bind_model"):
            raise TypeError("wsm_gradient_mtl_callback requires trainer.model and loss.bind_model")
        loss.bind_model(model)
        task_aware = getattr(model, "task_aware_fusion", None)
        if not isinstance(task_aware, bool):
            raise ValueError("gradient MTL model must expose boolean task_aware_fusion")
        expected = (
            {"shared": 14, "depression": 18, "parkinson": 18, "row": 1}
            if task_aware
            else {"shared": 23, "depression": 14, "parkinson": 14, "row": 0}
        )
        if loss.ownership_counts != expected:
            raise ValueError(
                f"unexpected structural ownership for task_aware_fusion={task_aware}: "
                f"{loss.ownership_counts}"
            )

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        self._loss(trainer).reset_epoch_diagnostics()

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        loss = self._loss(trainer)
        diagnostics = loss.epoch_diagnostics()
        logs.update(diagnostics)
        weights = getattr(loss, "gradnorm_weights", None)
        if loss.method == "gradnorm" and weights is not None:
            logs["mtl/final_weight_depression"] = float(weights[0])
            logs["mtl/final_weight_parkinson"] = float(weights[1])
        logger = getattr(trainer, "mlflow_logger", None)
        if logger is not None and diagnostics:
            logger.log_metrics(diagnostics, step=epoch)


@CALLBACKS.register("wsm_gradient_mtl_callback")
def wsm_gradient_mtl_callback(context: Any | None = None, **params: Any) -> WSMGradientMTLCallback:
    del context, params
    return WSMGradientMTLCallback()
