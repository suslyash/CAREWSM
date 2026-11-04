"""Corpus-specific validation monitor."""

from __future__ import annotations

import math
from typing import Any

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


class CAREWSMValidationMonitorCallback(BaseCallback):
    """Add the equal-weight corpus validation recall monitor to epoch logs."""

    components = (
        "val_care/depression/macro_recall",
        "val_care/parkinson/macro_recall",
        "val_wsm/depression/macro_recall",
        "val_wsm/parkinson/macro_recall",
    )

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        del epoch
        values = [logs.get(name) for name in self.components]
        invalid = [
            name
            for name, value in zip(self.components, values)
            if value is None or not math.isfinite(float(value))
        ]
        if invalid:
            raise ValueError(
                "dev_mean_macro_recall is unavailable; required validation "
                f"components are missing or invalid: {', '.join(invalid)}"
            )
        logs["dev_mean_macro_recall"] = sum(float(value) for value in values) / 4
        self._info(
            trainer,
            f"[validation monitor] dev_mean_macro_recall={logs['dev_mean_macro_recall']:.6f}",
        )


@CALLBACKS.register("carewsm_validation_monitor")
def carewsm_validation_monitor(
    context: Any | None = None,
    **params: Any,
) -> CAREWSMValidationMonitorCallback:
    del context, params
    return CAREWSMValidationMonitorCallback()
