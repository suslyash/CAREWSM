"""Compact per-epoch CAREWSM summary and TEST statistics."""

from __future__ import annotations

import csv
import io
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS
from chimera_ml.data.loader_utils import normalize_loaders

_TASKS = ("depression", "parkinson")
_METRICS = (
    "macro_f1",
    "weighted_f1",
    "macro_recall",
    "weighted_recall",
    "patient_recall",
    "healthy_recall",
)


class _TrainerEpochFilter(logging.Filter):
    """Suppress only stock Trainer epoch INFO lines in corpus mode."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            record.levelno == logging.INFO
            and any(
                message.startswith(f"[{split} epoch ")
                for split in ("train", "val_care", "val_wsm")
            )
        )


class _EvaluationLogger:
    """Suppress INFO during TEST while preserving warnings and errors."""

    def __init__(self, original: Any) -> None:
        self._original = original

    def info(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self._original.warning(*args, **kwargs)

    def error(self, *args: Any, **kwargs: Any) -> None:
        self._original.error(*args, **kwargs)

    def exception(self, *args: Any, **kwargs: Any) -> None:
        self._original.exception(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format(value: Any) -> str:
    number = _finite(value)
    return "-" if number is None else f"{number:.4f}"


@dataclass
class CAREWSMSummaryCallback(BaseCallback):
    """Evaluate fixed TEST after VAL and write compact per-epoch artifacts."""

    context: Any | None = field(default=None, repr=False)
    filename: str = "summary.txt"
    metrics_filename: str = "metrics_history.json"
    _path: Path | None = field(default=None, init=False)
    _metrics_path: Path | None = field(default=None, init=False)
    _rows: list[dict[str, Any]] = field(default_factory=list, init=False)
    _test_loaders: dict[str, Any] = field(default_factory=dict, init=False)
    _corpus_mode: bool = field(default=False, init=False)
    _epoch_filter: _TrainerEpochFilter | None = field(default=None, init=False)

    def on_fit_start(self, trainer: Any) -> None:
        log_path = getattr(getattr(trainer, "logger", None), "log_path", None)
        run_dir = Path(log_path).parent if log_path else Path("logs")
        self._path = run_dir / self.filename
        self._metrics_path = run_dir / self.metrics_filename
        self._path.parent.mkdir(parents=True, exist_ok=True)

        data_module = (
            self.context.get("data.module") if self.context is not None else None
        )
        self._corpus_mode = (
            getattr(data_module, "evaluation_layout", "legacy") == "corpus"
        )
        if data_module is not None:
            self._test_loaders = normalize_loaders(
                data_module.test_dataloader(),
                default_name="test",
            )
        if self._corpus_mode:
            self._install_epoch_filter(trainer)

        self._write()

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        try:
            test_logs = self._evaluate_test(trainer, epoch)
        except Exception:
            self._remove_epoch_filter(trainer)
            raise
        if self._corpus_mode:
            self._on_corpus_epoch_end(trainer, epoch, logs, test_logs)
            return
        combined = dict(logs)
        combined.update(test_logs)
        checkpoint = self._checkpoint_for_epoch(trainer, epoch)

        record = {
            "epoch": epoch,
            "checkpoint": checkpoint,
            "logs": {
                key: (
                    float(value)
                    if isinstance(value, (int, float)) and math.isfinite(float(value))
                    else None
                )
                for key, value in combined.items()
                if isinstance(value, (int, float))
            },
        }
        self._rows.append(record)
        self._write()
        self._log_compact_epoch(trainer, record)

    def _evaluate_test(self, trainer: Any, epoch: int) -> dict[str, float]:
        """Evaluate TEST with no callback side effects.

        Chimera-ML 0.3.0 exposes no per-epoch TEST hook. Its public
        evaluate method executes all callbacks, so this callback uses the
        private epoch primitive and restores the logger in a finally block.
        """
        if not self._test_loaders:
            return {}

        device = torch.device(
            trainer.config.device if torch.cuda.is_available() else "cpu"
        )
        scaler = torch.amp.GradScaler(device=device.type, enabled=False)
        original_logger = trainer.logger
        trainer.logger = _EvaluationLogger(original_logger)
        results: dict[str, float] = {}
        try:
            for name, loader in self._test_loaders.items():
                split = (
                    name
                    if self._corpus_mode and name.startswith("test_")
                    else f"test_{name.lower()}"
                )
                metrics = trainer._run_epoch(
                    loader=loader,
                    device=device,
                    scaler=scaler,
                    train=False,
                    epoch=epoch,
                    split=split,
                )
                prefix = name if self._corpus_mode else f"test/{name}"
                results.update(
                    {f"{prefix}/{key}": value for key, value in metrics.items()}
                )
        finally:
            trainer.logger = original_logger

        self._add_test_averages(results)
        if trainer.mlflow_logger:
            mlflow_values = {
                key: float(value)
                for key, value in results.items()
                if _finite(value) is not None
            }
            trainer.mlflow_logger.log_metrics(mlflow_values, step=epoch)
        return results

    @staticmethod
    def _add_test_averages(results: dict[str, float]) -> None:
        for task in _TASKS:
            care_count = results.get(
                f"test/CARE/care/{task}/observed_count",
                0.0,
            )
            wsm_count = results.get(
                f"test/WSM/wsm/{task}/observed_count",
                0.0,
            )
            for metric in _METRICS:
                care_value = results.get(
                    f"test/CARE/care/{task}/{metric}",
                    float("nan"),
                )
                wsm_value = results.get(
                    f"test/WSM/wsm/{task}/{metric}",
                    float("nan"),
                )
                results[f"test/average/{task}/{metric}"] = (
                    (care_value + wsm_value) / 2
                    if care_count
                    and wsm_count
                    and _finite(care_value) is not None
                    and _finite(wsm_value) is not None
                    else float("nan")
                )
        care_loss = results.get("test/CARE/loss", float("nan"))
        wsm_loss = results.get("test/WSM/loss", float("nan"))
        results["test/average/loss"] = (
            (care_loss + wsm_loss) / 2
            if _finite(care_loss) is not None and _finite(wsm_loss) is not None
            else float("nan")
        )

    def _checkpoint_for_epoch(self, trainer: Any, epoch: int) -> str | None:
        for callback in trainer.callbacks:
            if getattr(callback, "monitor", None) != "joint/average_macro_f1":
                continue
            for path in getattr(callback, "_saved", []):
                if Path(path).name.startswith(f"epoch={epoch}_"):
                    return str(path)
        return None

    def _log_compact_epoch(self, trainer: Any, record: dict[str, Any]) -> None:
        values = record["logs"]
        pieces = [
            f"epoch={record['epoch']}",
            f"train_loss={_format(values.get('train/loss'))}",
            f"val_macro_f1=({_format(values.get('joint/care/depression/macro_f1'))},"
            f"{_format(values.get('joint/wsm/depression/macro_f1'))};"
            f"{_format(values.get('joint/care/parkinson/macro_f1'))},"
            f"{_format(values.get('joint/wsm/parkinson/macro_f1'))})",
            "val_patient_recall=("
            f"{_format(values.get('joint/care/depression/patient_recall'))},"
            f"{_format(values.get('joint/wsm/depression/patient_recall'))};"
            f"{_format(values.get('joint/care/parkinson/patient_recall'))},"
            f"{_format(values.get('joint/wsm/parkinson/patient_recall'))})",
            f"val_monitor={_format(values.get('joint/average_macro_f1'))}",
            "test_macro_f1=("
            f"{_format(values.get('test/CARE/care/depression/macro_f1'))},"
            f"{_format(values.get('test/WSM/wsm/depression/macro_f1'))};"
            f"{_format(values.get('test/CARE/care/parkinson/macro_f1'))},"
            f"{_format(values.get('test/WSM/wsm/parkinson/macro_f1'))})",
        ]
        trainer.logger.warning("[epoch summary] " + " ".join(pieces))


    def _install_epoch_filter(self, trainer: Any) -> None:
        logger = getattr(trainer.logger, "logger", trainer.logger)
        handlers = getattr(logger, "handlers", [])
        self._epoch_filter = _TrainerEpochFilter()
        for handler in handlers:
            handler.addFilter(self._epoch_filter)

    def _remove_epoch_filter(self, trainer: Any) -> None:
        if self._epoch_filter is None:
            return
        logger = getattr(trainer.logger, "logger", trainer.logger)
        for handler in getattr(logger, "handlers", []):
            handler.removeFilter(self._epoch_filter)
        self._epoch_filter = None

    def on_fit_end(self, trainer: Any) -> None:
        self._remove_epoch_filter(trainer)

    def _on_corpus_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        logs: dict[str, Any],
        test_logs: dict[str, float],
    ) -> None:
        combined = dict(logs)
        combined.update(test_logs)
        record = {"epoch": epoch, "logs": combined}
        self._rows = [row for row in self._rows if row.get("epoch") != epoch]
        self._rows.append(record)
        self._write_corpus_summary()
        self._log_corpus_lines(trainer, epoch, combined)

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

    def _log_corpus_lines(
        self,
        trainer: Any,
        epoch: int,
        logs: dict[str, Any],
    ) -> None:
        for population in ("train", "val_care", "val_wsm", "test_care", "test_wsm"):
            values = [f"epoch={epoch}", f"split={population}"]
            values.append(f"loss={self._text_value(logs.get(f'{population}/loss'))}")
            values.append(
                f"num_samples={self._count_text(logs.get(f'{population}/num_samples'))}"
            )
            for task in _TASKS:
                prefix = f"{population}/{task}"
                values.extend(
                    [
                        (
                            f"{task}_observed="
                            f"{self._count_text(logs.get(f'{prefix}/observed_count'))}"
                        ),
                        (
                            f"{task}_macro_recall="
                            f"{self._text_value(logs.get(f'{prefix}/macro_recall'))}"
                        ),
                        (
                            f"{task}_patient_recall="
                            f"{self._text_value(logs.get(f'{prefix}/patient_recall'))}"
                        ),
                        (
                            f"{task}_healthy_recall="
                            f"{self._text_value(logs.get(f'{prefix}/healthy_recall'))}"
                        ),
                    ]
                )
            trainer.logger.info("[epoch summary] " + " ".join(values))

    def _write(self) -> None:
        if self._corpus_mode:
            self._write_corpus_summary()
            return
        if self._path is None or self._metrics_path is None:
            return

        self._metrics_path.write_text(
            json.dumps(self._rows, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        fieldnames = [
            "epoch",
            "split",
            "task",
            "loss",
            *_METRICS,
            "CARE_count",
            "WSM_count",
            "total_count",
        ]
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for record in self._rows:
            for split, prefix, loss_prefix, care_prefix, wsm_prefix in (
                self._table_splits()
            ):
                values = record["logs"]
                for task in _TASKS:
                    loss = (
                        values.get(f"{loss_prefix}/loss")
                        if task == "depression"
                        else None
                    )
                    care_count = self._count(values, care_prefix, task)
                    wsm_count = self._count(values, wsm_prefix, task)
                    row = {
                        "epoch": record["epoch"],
                        "split": split,
                        "task": task,
                        "loss": _format(loss) if loss is not None else "",
                        "CARE_count": _format(care_count),
                        "WSM_count": _format(wsm_count),
                        "total_count": _format(care_count + wsm_count),
                    }
                    row.update(
                        {
                            metric: _format(values.get(f"{prefix}/{task}/{metric}"))
                            for metric in _METRICS
                        }
                    )
                    writer.writerow(row)

        self._path.write_text(output.getvalue(), encoding="utf-8")

    @staticmethod
    def _count(values: dict[str, Any], prefix: str | None, task: str) -> float:
        if prefix is None:
            return 0.0
        return float(values.get(f"{prefix}/{task}/observed_count") or 0.0)

    @staticmethod
    def _table_splits() -> list[tuple[str, str, str, str | None, str | None]]:
        return [
            ("TRAIN", "train/average", "train", "train/care", "train/wsm"),
            ("VAL CARE", "joint/care", "joint", "joint/care", None),
            ("VAL WSM", "joint/wsm", "joint", None, "joint/wsm"),
            ("VAL average", "joint/average", "joint", "joint/care", "joint/wsm"),
            ("TEST CARE", "test/CARE/care", "test/CARE", "test/CARE/care", None),
            ("TEST WSM", "test/WSM/wsm", "test/WSM", None, "test/WSM/wsm"),
            (
                "TEST average",
                "test/average",
                "test/average",
                "test/CARE/care",
                "test/WSM/wsm",
            ),
        ]

    def _best_row(self) -> dict[str, Any] | None:
        candidates = [
            row
            for row in self._rows
            if _finite(row["logs"].get("joint/average_macro_f1")) is not None
        ]
        return max(
            candidates,
            key=lambda row: float(row["logs"]["joint/average_macro_f1"]),
            default=None,
        )

    def _run_metadata(self) -> str:
        config = self.context.config if self.context is not None else None
        experiment = (
            config.section("experiment_info").get("params", {}) if config else {}
        )
        data = config.section("data").get("params", {}) if config else {}
        model_section = config.section("model") if config else {}
        seed = config.get("seed", "unknown") if config else "unknown"
        model = model_section.get("name", "unknown")
        family = data.get("feature_family", "unknown")
        fold = data.get("fold", "unknown")
        return (
            f"model: {model} | feature family: {family} | fold: {fold} | "
            f"seed: {seed} | experiment: {experiment.get('experiment_name', 'unknown')}"
        )


CarewsmSummaryCallback = CAREWSMSummaryCallback


@CALLBACKS.register("carewsm_summary_callback")
def carewsm_summary_callback(
    context: Any | None = None,
    filename: str = "summary.txt",
    metrics_filename: str = "metrics_history.json",
    **params: Any,
) -> CAREWSMSummaryCallback:
    del params
    return CAREWSMSummaryCallback(
        context=context,
        filename=filename,
        metrics_filename=metrics_filename,
    )
