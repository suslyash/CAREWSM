"""DEV-only controller callback for the frozen R4 task-balancing study."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


@dataclass
class WSMR4BalanceCallback(BaseCallback):
    """Log the completed epoch controller state, then update the next epoch."""

    def on_fit_start(self, trainer: Any) -> None:
        loss = getattr(trainer, "loss_fn", None)
        if loss is None or not hasattr(loss, "update_controller") or not hasattr(loss, "weights"):
            raise TypeError("wsm_r4_balance_callback requires the R4 controller loss")
        if getattr(loss, "mode", None) not in {"equal", "stch", "progress", "ra_stch"}:
            raise ValueError("unsupported R4 controller mode")
        self._assert_weights(loss.weights)

    @staticmethod
    def _assert_weights(weights: Any) -> None:
        values = [float(value) for value in weights]
        if len(values) != 2 or not all(math.isfinite(value) for value in values) or any(value < 0 or value > 1 for value in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError("R4 controller weights must be finite, bounded, and normalized")

    @staticmethod
    def _put(logs: dict[str, float], name: str, value: Any) -> None:
        if value is not None and math.isfinite(float(value)):
            logs[name] = float(value)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        loss = trainer.loss_fn
        self._assert_weights(loss.weights)
        used = [float(value) for value in loss.weights]
        logs["r4/weight_depression"], logs["r4/weight_parkinson"] = used
        self._put(logs, "r4/progress_depression", loss.last_progress[0])
        self._put(logs, "r4/progress_parkinson", loss.last_progress[1])
        self._put(logs, "r4/grad_norm_depression", loss.grad_norm_ema[0])
        self._put(logs, "r4/grad_norm_parkinson", loss.grad_norm_ema[1])
        self._put(logs, "r4/grad_cosine", loss.grad_cos_ema)
        self._put(logs, "r4/reliability_depression", loss.reliability_ema[0] if loss.reliability_ema[0] is not None else 0.5)
        self._put(logs, "r4/reliability_parkinson", loss.reliability_ema[1] if loss.reliability_ema[1] is not None else 0.5)
        if "dev/depression/score" not in logs or "dev/parkinson/score" not in logs:
            raise KeyError("R4 controller requires DEV task scores and never consumes Test metrics")
        trainer_loss_logger = getattr(trainer, "mlflow_logger", None)
        if trainer_loss_logger is not None:
            trainer_loss_logger.log_metrics({key: value for key, value in logs.items() if key.startswith("r4/")}, step=epoch)
        loss.update_controller(logs["dev/depression/score"], logs["dev/parkinson/score"])
        self._assert_weights(loss.weights)


@CALLBACKS.register("wsm_r4_balance_callback")
def wsm_r4_balance_callback(context: Any | None = None, **params: Any) -> WSMR4BalanceCallback:
    del context
    return WSMR4BalanceCallback(**params)
