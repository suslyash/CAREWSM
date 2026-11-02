"""VAL-only checkpoint and early-stopping callbacks."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from chimera_ml.callbacks.checkpoint_callback import CheckpointCallback
from chimera_ml.callbacks.early_stopping_callback import EarlyStoppingCallback
from chimera_ml.core.registry import CALLBACKS


class CarewsmValCheckpointCallback(CheckpointCallback):
    """Do not save a best checkpoint when the VAL monitor is undefined."""

    def on_fit_start(self, trainer: Any) -> None:
        """Place checkpoints beside the logger's other run artifacts.

        Chimera-ML's CLI injects runtime names into its built-in
        ``checkpoint_callback`` only. This project-specific callback is
        intentionally named differently, so the active console logger is the
        authoritative source for the run directory.
        """
        log_file = getattr(getattr(trainer, "logger", None), "log_path", None)
        if log_file is not None:
            self._resolved_dirpath = Path(log_file).parent / "checkpoints"
            self._resolved_dirpath.mkdir(parents=True, exist_ok=True)
            return

        super().on_fit_start(trainer)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        value = logs.get(self.monitor)
        if value is not None and not math.isfinite(float(value)):
            if self.save_last:
                self._save(trainer, epoch, trainer.global_step, float("nan"), is_last=True)
            self._warning(trainer, f"Skipping non-finite VAL checkpoint monitor {self.monitor}")
            return
        super().on_epoch_end(trainer, epoch, logs)


class CarewsmValEarlyStoppingCallback(EarlyStoppingCallback):
    """Ignore undefined VAL monitors instead of treating them as improvements."""

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        value = logs.get(self.monitor)
        if value is not None and not math.isfinite(float(value)):
            self._warning(trainer, f"Skipping non-finite VAL early-stopping monitor {self.monitor}")
            return
        super().on_epoch_end(trainer, epoch, logs)


@CALLBACKS.register("carewsm_val_checkpoint_callback")
def carewsm_val_checkpoint_callback(**params: Any) -> CarewsmValCheckpointCallback:
    return CarewsmValCheckpointCallback(**params)


@CALLBACKS.register("carewsm_val_early_stopping_callback")
def carewsm_val_early_stopping_callback(**params: Any) -> CarewsmValEarlyStoppingCallback:
    return CarewsmValEarlyStoppingCallback(**params)
