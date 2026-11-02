"""VAL-only checkpoint and early-stopping callbacks."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from chimera_ml.callbacks.checkpoint_callback import CheckpointCallback
from chimera_ml.callbacks.early_stopping_callback import EarlyStoppingCallback
from chimera_ml.core.registry import CALLBACKS
from chimera_ml.logging.console_file_logger import ConsoleFileLogger


class CarewsmValCheckpointCallback(CheckpointCallback):
    """Do not save a best checkpoint when the VAL monitor is undefined."""

    def on_fit_start(self, trainer: Any) -> None:
        """Place checkpoints beside the logger's other run artifacts.

        Chimera-ML 0.3.0 creates a ``ConsoleFileLogger`` before callbacks and
        stores the concrete ``train.log`` path in its ``log_path`` attribute.
        The parent of that path is the authoritative directory for this run.
        """
        logger = getattr(trainer, "logger", None)
        if not isinstance(logger, ConsoleFileLogger):
            raise RuntimeError(
                "CarewsmValCheckpointCallback requires Chimera-ML's "
                "ConsoleFileLogger on trainer.logger; the current run "
                "directory cannot be determined safely."
            )

        log_file = logger.log_path
        if not isinstance(log_file, Path) or log_file.name != "train.log":
            raise RuntimeError(
                "CarewsmValCheckpointCallback requires logger.log_path to "
                "be a pathlib.Path naming the current run's train.log; "
                f"received {log_file!r}."
            )

        run_dir = log_file.parent
        if run_dir == log_file or run_dir == Path("."):
            raise RuntimeError(
                "CarewsmValCheckpointCallback could not resolve a distinct "
                f"run directory from logger.log_path={log_file!s}."
            )

        self._resolved_dirpath = run_dir / "checkpoints"
        self._resolved_dirpath.mkdir(parents=True, exist_ok=True)

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
