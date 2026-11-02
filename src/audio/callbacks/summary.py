"""Readable per-run epoch summary callback."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


@dataclass
class CarewsmSummaryCallback(BaseCallback):
    filename: str = "summary.txt"
    _path: Path | None = field(default=None, init=False)
    _rows: list[dict[str, Any]] = field(default_factory=list, init=False)

    def on_fit_start(self, trainer: Any) -> None:
        log_path = getattr(getattr(trainer, "logger", None), "log_path", None)
        self._path = (Path(log_path).parent if log_path else Path("logs")) / self.filename
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write()

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        del trainer
        row = {"epoch": epoch}
        row.update({key: value for key, value in logs.items() if isinstance(value, (int, float))})
        self._rows.append(row)
        self._write()

    def _write(self) -> None:
        if self._path is None:
            return
        keys = sorted({key for row in self._rows for key in row if key != "epoch"})
        lines = ["epoch;" + ";".join(keys)]
        for row in self._rows:
            values = [str(row["epoch"])]
            for key in keys:
                value = row.get(key)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    values.append("UNDEFINED")
                else:
                    values.append(f"{float(value):.6f}")
            lines.append(";".join(values))
        self._path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@CALLBACKS.register("carewsm_summary_callback")
def carewsm_summary_callback(context: Any | None = None, filename: str = "summary.txt", **params: Any) -> CarewsmSummaryCallback:
    return CarewsmSummaryCallback(filename=filename)
