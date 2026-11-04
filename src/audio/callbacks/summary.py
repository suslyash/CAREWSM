"""Wide per-epoch CAREWSM summary and corpus TEST evaluation."""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

import torch

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS

_TASKS = ("depression", "parkinson")
_METRICS = (
    "macro_f1",
    "weighted_f1",
    "macro_recall",
    "weighted_recall",
    "patient_recall",
    "healthy_recall",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class CAREWSMSummaryCallback(BaseCallback):
    """Evaluate fixed TEST after VAL and write one wide row per epoch."""

    def __init__(
        self, context: Any | None = None, filename: str = "summary.txt"
    ) -> None:
        self.context = context
        self.filename = filename
        self._path: Path | None = None
        self._rows: list[dict[str, Any]] = []
        self._test_loaders: dict[str, Any] = {}
        self._corpus_test_results: dict[int, dict[str, float]] = {}
        self._completed_corpus_epochs: set[int] = set()

    def on_fit_start(self, trainer: Any) -> None:
        log_path = getattr(getattr(trainer, "logger", None), "log_path", None)
        run_dir = Path(log_path).parent if log_path else Path("logs")
        self._path = run_dir / self.filename
        self._path.parent.mkdir(parents=True, exist_ok=True)

        data_module = (
            self.context.get("data.module") if self.context is not None else None
        )
        if data_module is not None:
            self._test_loaders = dict(data_module.test_dataloader())
            if set(self._test_loaders) != {"test_care", "test_wsm"}:
                raise ValueError(
                    "DataModule must provide test_care and test_wsm loaders"
                )
        self._write_corpus_summary()

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        if epoch in self._completed_corpus_epochs:
            combined = dict(logs)
            combined.update(self._corpus_test_results[epoch])
            self._replace_corpus_row(epoch, combined)
            self._write_corpus_summary()
            return

        test_logs = self._evaluate_test(trainer, epoch)
        self._corpus_test_results[epoch] = dict(test_logs)
        combined = dict(logs)
        combined.update(test_logs)
        self._replace_corpus_row(epoch, combined)
        self._write_corpus_summary()
        self._completed_corpus_epochs.add(epoch)

    def _evaluate_test(self, trainer: Any, epoch: int) -> dict[str, float]:
        """Evaluate TEST with no callback side effects.

        Chimera-ML 0.3.0 exposes no per-epoch TEST hook. Its public
        evaluate method executes all callbacks, so this callback uses the
        private epoch primitive and keeps the Trainer logger unchanged.
        """
        if not self._test_loaders:
            return {}

        device = torch.device(
            trainer.config.device if torch.cuda.is_available() else "cpu"
        )
        scaler = torch.amp.GradScaler(device=device.type, enabled=False)
        results: dict[str, float] = {}
        for name, loader in self._test_loaders.items():
            split = name
            metrics = trainer._run_epoch(
                loader=loader,
                device=device,
                scaler=scaler,
                train=False,
                epoch=epoch,
                split=split,
            )
            prefix = name
            results.update(
                {f"{prefix}/{key}": value for key, value in metrics.items()}
            )

        if getattr(trainer, "mlflow_logger", None) is not None:
            mlflow_values = {
                key: float(value)
                for key, value in results.items()
                if _finite(value) is not None
            }
            trainer.mlflow_logger.log_metrics(mlflow_values, step=epoch)
        return results

    def _replace_corpus_row(self, epoch: int, logs: dict[str, Any]) -> None:
        record = {"epoch": epoch, "logs": logs}
        self._rows = [row for row in self._rows if row.get("epoch") != epoch]
        self._rows.append(record)

    @staticmethod
    def _corpus_columns() -> list[str]:
        columns = ["epoch"]
        for population in ("train", "val_care", "val_wsm", "test_care", "test_wsm"):
            columns.extend([f"{population}_loss", f"{population}_num_samples"])
            for task in _TASKS:
                columns.append(f"{population}_{task}_observed_count")
                columns.append(f"{population}_{task}_positive_support")
                columns.append(f"{population}_{task}_negative_support")
                columns.extend(
                    f"{population}_{task}_{metric}" for metric in _METRICS
                )
        columns.append("dev_mean_macro_recall")
        return columns

    def _write_corpus_summary(self) -> None:
        if self._path is None:
            return
        columns = self._corpus_columns()
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output,
            fieldnames=columns,
            delimiter=";",
            lineterminator="\n",
        )
        writer.writeheader()
        for record in self._rows:
            logs = record["logs"]
            row = {"epoch": record["epoch"]}
            for population in ("train", "val_care", "val_wsm", "test_care", "test_wsm"):
                row[f"{population}_loss"] = self._text_value(
                    logs.get(f"{population}/loss")
                )
                row[f"{population}_num_samples"] = self._count_text(
                    logs.get(f"{population}/num_samples")
                )
                for task in _TASKS:
                    prefix = f"{population}/{task}"
                    for name in (
                        "observed_count",
                        "positive_support",
                        "negative_support",
                    ):
                        row[f"{population}_{task}_{name}"] = self._count_text(
                            logs.get(f"{prefix}/{name}")
                        )
                    for metric in _METRICS:
                        row[f"{population}_{task}_{metric}"] = self._text_value(
                            logs.get(f"{prefix}/{metric}")
                        )
            row["dev_mean_macro_recall"] = self._text_value(
                logs.get("dev_mean_macro_recall")
            )
            writer.writerow(row)
        self._path.write_text(output.getvalue(), encoding="utf-8")

    @staticmethod
    def _text_value(value: Any) -> str:
        number = _finite(value)
        return "-" if number is None else f"{number:.4f}"

    @staticmethod
    def _count_text(value: Any) -> str:
        number = _finite(value)
        return "-" if number is None else str(int(number))


@CALLBACKS.register("carewsm_summary_callback")
def carewsm_summary_callback(
    context: Any | None = None,
    filename: str = "summary.txt",
    **params: Any,
) -> CAREWSMSummaryCallback:
    del params
    return CAREWSMSummaryCallback(
        context=context,
        filename=filename,
    )
