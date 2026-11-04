"""Wide per-epoch CAREWSM summary persistence."""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


def _text(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return str(int(number)) if number.is_integer() else f"{number:.4f}"


class CAREWSMSummaryCallback(BaseCallback):
    """Write Chimera's completed epoch logs to one wide summary row."""

    def __init__(self, filename: str = "summary.txt") -> None:
        self.filename = filename
        self._path: Path | None = None
        self._rows: dict[int, dict[str, Any]] = {}
        self._columns: list[str] = ["epoch"]

    def on_fit_start(self, trainer: Any) -> None:
        log_path = getattr(getattr(trainer, "logger", None), "log_path", None)
        run_dir = Path(log_path).parent if log_path else Path("logs")
        self._path = run_dir / self.filename
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write()

    def on_epoch_end(
        self, trainer: Any, epoch: int, logs: dict[str, Any]
    ) -> None:
        del trainer
        row = {
            key.replace("/", "_"): value
            for key, value in logs.items()
            if isinstance(value, (int, float))
        }
        for key in sorted(row):
            if key not in self._columns:
                self._columns.append(key)
        self._rows[epoch] = row
        self._write()

    def _write(self) -> None:
        if self._path is None:
            return
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output,
            fieldnames=self._columns,
            delimiter=";",
            lineterminator="\n",
        )
        writer.writeheader()
        for epoch in sorted(self._rows):
            values = self._rows[epoch]
            writer.writerow(
                {
                    "epoch": epoch,
                    **{key: _text(values.get(key)) for key in self._columns[1:]},
                }
            )
        self._path.write_text(output.getvalue(), encoding="utf-8")


@CALLBACKS.register("carewsm_summary_callback")
def carewsm_summary_callback(
    context: Any | None = None,
    filename: str = "summary.txt",
    **params: Any,
) -> CAREWSMSummaryCallback:
    del context, params
    return CAREWSMSummaryCallback(filename=filename)
