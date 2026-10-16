"""Frozen pseudo-supervision scale warm-up callback."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


@dataclass
class WSMPseudoScaleWarmupCallback(BaseCallback):
    """Set the observed-to-pseudo supervision scale on one-based epoch starts."""

    observed_only_epochs: int = 3
    ramp_epochs: int = 5
    final_scale: float = 1.0
    current_scale: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.observed_only_epochs, Integral) or isinstance(self.observed_only_epochs, bool) or self.observed_only_epochs < 0:
            raise ValueError("observed_only_epochs must be an integer >= 0")
        if not isinstance(self.ramp_epochs, Integral) or isinstance(self.ramp_epochs, bool) or self.ramp_epochs < 1:
            raise ValueError("ramp_epochs must be an integer >= 1")
        if not isinstance(self.final_scale, Real) or not math.isfinite(float(self.final_scale)) or not 0.0 <= float(self.final_scale) <= 1.0:
            raise ValueError("final_scale must be finite and in [0, 1]")
        self.observed_only_epochs = int(self.observed_only_epochs)
        self.ramp_epochs = int(self.ramp_epochs)
        self.final_scale = float(self.final_scale)

    def scale_for_epoch(self, epoch: int) -> float:
        if not isinstance(epoch, Integral) or isinstance(epoch, bool) or epoch < 1:
            raise ValueError("epoch must be an integer >= 1")
        if epoch <= self.observed_only_epochs:
            return 0.0
        progress = min(1.0, (epoch - self.observed_only_epochs) / self.ramp_epochs)
        return float(self.final_scale * progress)

    def on_fit_start(self, trainer: Any) -> None:
        loss_fn = getattr(trainer, "loss_fn", None)
        if loss_fn is None or not hasattr(loss_fn, "pseudo_scale"):
            raise TypeError("warm-up requires trainer.loss_fn.pseudo_scale")
        current = getattr(loss_fn, "pseudo_scale")
        if not isinstance(current, Real) or isinstance(current, bool) or not math.isfinite(float(current)) or not 0.0 <= float(current) <= 1.0:
            raise ValueError("trainer.loss_fn.pseudo_scale must be finite and in [0, 1]")
        try:
            loss_fn.pseudo_scale = self.scale_for_epoch(1)
        except Exception as exc:
            raise TypeError("trainer.loss_fn.pseudo_scale must be writable") from exc
        if float(loss_fn.pseudo_scale) != 0.0:
            raise RuntimeError("warm-up failed to initialize epoch-1 pseudo_scale=0.0")
        self.current_scale = 0.0

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        loss_fn = getattr(trainer, "loss_fn", None)
        if loss_fn is None or not hasattr(loss_fn, "pseudo_scale"):
            raise TypeError("warm-up requires trainer.loss_fn.pseudo_scale")
        scale = self.scale_for_epoch(epoch)
        try:
            loss_fn.pseudo_scale = scale
        except Exception as exc:
            raise TypeError("trainer.loss_fn.pseudo_scale must be writable") from exc
        if not math.isclose(float(loss_fn.pseudo_scale), scale, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("trainer.loss_fn.pseudo_scale did not accept the scheduled value")
        self.current_scale = scale

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        if self.current_scale is None:
            raise RuntimeError("warm-up epoch end called before epoch start")
        logs["train/pseudo_scale"] = float(self.current_scale)
        mlflow_logger = getattr(trainer, "mlflow_logger", None)
        if mlflow_logger is not None:
            mlflow_logger.log_metrics({"train/pseudo_scale": float(self.current_scale)}, step=epoch)


@CALLBACKS.register("wsm_pseudo_scale_warmup_callback")
def wsm_pseudo_scale_warmup_callback(context: Any | None = None, **params: Any) -> WSMPseudoScaleWarmupCallback:
    del context
    return WSMPseudoScaleWarmupCallback(**params)
