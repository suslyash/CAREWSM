from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import math

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


@dataclass
class WSMSummaryCallback(BaseCallback):
    """Write numeric epoch metrics to a compact TXT file."""

    filename: str = "summary.txt"

    _history: list[dict[str, float]] = field(default_factory=list, init=False)
    _path: Path | None = field(default=None, init=False)

    def on_fit_start(self, trainer: Any) -> None:
        log_path = getattr(getattr(trainer, "logger", None), "log_path", None)
        self._path = (Path(log_path).parent if log_path is not None else Path("logs")) / self.filename
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write()

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        row = {
            key: float(value)
            for key, value in logs.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        row["epoch"] = float(epoch)
        self._history.append(row)
        self._write()

    def _write(self) -> None:
        if self._path is None:
            return

        if not self._history:
            self._path.write_text("epoch\n", encoding="utf-8")
            return

        keys = self._metric_keys()
        lines = [";".join(["epoch", *keys])]
        for row in self._history:
            values = [str(int(row["epoch"]))]
            values.extend(self._format(row.get(key)) for key in keys)
            lines.append(";".join(values))

        self._path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _metric_keys(self) -> list[str]:
        keys = sorted({key for row in self._history for key in row if key != "epoch"})
        return sorted(keys, key=self._sort_key)

    @staticmethod
    def _sort_key(key: str) -> tuple[int, int, int, str]:
        prefix = key.split("/", 1)[0]
        split_priority = {
            "train": 0,
            "dev": 1,
            "dev_depression": 2,
            "dev_parkinson": 3,
            "test_none": 4,
            "test_none_depression": 5,
            "test_none_parkinson": 6,
            "test_soft": 7,
            "test_soft_depression": 8,
            "test_soft_parkinson": 9,
            "test_hard": 10,
            "test_hard_depression": 11,
            "test_hard_parkinson": 12,
        }
        metric = key.rsplit("/", 1)[-1]
        metric_priority = {
            "num_samples": 0,
            "loss": 1,
            "mean_loss": 2,
            "macro_precision": 3,
            "mean_macro_precision": 4,
            "macro_recall": 5,
            "mean_macro_recall": 6,
            "macro_f1": 7,
            "mean_macro_f1": 8,
            "cm_acc": 9,
        }
        task_priority = 0
        if "/depression/" in key or key.startswith("dev_depression/") or "_depression/" in key:
            task_priority = 1
        elif "/parkinson/" in key or key.startswith("dev_parkinson/") or "_parkinson/" in key:
            task_priority = 2

        return (split_priority.get(prefix, 100), task_priority, metric_priority.get(metric, 100), key)

    @staticmethod
    def _format(value: float | None) -> str:
        return "" if value is None else f"{value:.6f}"


@CALLBACKS.register("wsm_summary_callback")
def wsm_summary_callback(context: Any | None = None, filename: str = "summary.txt") -> WSMSummaryCallback:
    return WSMSummaryCallback(filename=filename)
