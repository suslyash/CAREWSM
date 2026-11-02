"""Compact per-epoch CAREWSM summary and TEST statistics."""

from __future__ import annotations

import json
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
class CarewsmSummaryCallback(BaseCallback):
    """Evaluate fixed TEST after VAL and write compact per-epoch artifacts."""

    context: Any | None = field(default=None, repr=False)
    filename: str = "summary.txt"
    metrics_filename: str = "metrics_history.json"
    _path: Path | None = field(default=None, init=False)
    _metrics_path: Path | None = field(default=None, init=False)
    _rows: list[dict[str, Any]] = field(default_factory=list, init=False)
    _test_loaders: dict[str, Any] = field(default_factory=dict, init=False)

    def on_fit_start(self, trainer: Any) -> None:
        log_path = getattr(getattr(trainer, "logger", None), "log_path", None)
        run_dir = Path(log_path).parent if log_path else Path("logs")
        self._path = run_dir / self.filename
        self._metrics_path = run_dir / self.metrics_filename
        self._path.parent.mkdir(parents=True, exist_ok=True)

        data_module = self.context.get("data.module") if self.context is not None else None
        if data_module is not None:
            self._test_loaders = normalize_loaders(
                data_module.test_dataloader(),
                default_name="test",
            )

        self._write()

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        test_logs = self._evaluate_test(trainer, epoch)
        combined = dict(logs)
        combined.update(test_logs)
        combined = {
            key: value
            for key, value in combined.items()
            if not ("/average/" in key and key.endswith("/observed_count"))
            and not (key.startswith("average/") and key.endswith("/observed_count"))
        }
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

        device = torch.device(trainer.config.device if torch.cuda.is_available() else "cpu")
        scaler = torch.amp.GradScaler(device=device.type, enabled=False)
        original_logger = trainer.logger
        trainer.logger = _EvaluationLogger(original_logger)
        results: dict[str, float] = {}
        try:
            for name, loader in self._test_loaders.items():
                metrics = trainer._run_epoch(
                    loader=loader,
                    device=device,
                    scaler=scaler,
                    train=False,
                    epoch=epoch,
                    split=f"test_{name}",
                )
                results.update(
                    {f"test/{name}/{key}": value for key, value in metrics.items()}
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
            f"val_macro_f1=({_format(values.get('CARE/care/depression/macro_f1'))},"
            f"{_format(values.get('WSM/wsm/depression/macro_f1'))};"
            f"{_format(values.get('CARE/care/parkinson/macro_f1'))},"
            f"{_format(values.get('WSM/wsm/parkinson/macro_f1'))})",
            f"val_patient_recall=({_format(values.get('CARE/care/depression/patient_recall'))},"
            f"{_format(values.get('WSM/wsm/depression/patient_recall'))};"
            f"{_format(values.get('CARE/care/parkinson/patient_recall'))},"
            f"{_format(values.get('WSM/wsm/parkinson/patient_recall'))})",
            f"val_monitor={_format(values.get('joint/average_macro_f1'))}",
            f"test_macro_f1=({_format(values.get('test/CARE/care/depression/macro_f1'))},"
            f"{_format(values.get('test/WSM/wsm/depression/macro_f1'))};"
            f"{_format(values.get('test/CARE/care/parkinson/macro_f1'))},"
            f"{_format(values.get('test/WSM/wsm/parkinson/macro_f1'))})",
        ]
        trainer.logger.warning("[epoch summary] " + " ".join(pieces))

    def _write(self) -> None:
        if self._path is None or self._metrics_path is None:
            return

        self._metrics_path.write_text(
            json.dumps(self._rows, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        best = self._best_row()
        lines = [
            "CAREWSM epoch summary",
            "=====================",
            self._run_metadata(),
            "",
            "Best VAL epoch: " + (str(best["epoch"]) if best else "-"),
            "Best VAL monitor (joint/average_macro_f1): "
            + (_format(best["logs"].get("joint/average_macro_f1")) if best else "-"),
            "Best VAL checkpoint: "
            + (best.get("checkpoint") or "-" if best else "-"),
            "",
            "epoch | split | task | loss | macro_f1 | weighted_f1 | macro_recall | "
            "weighted_recall | patient_recall | healthy_recall | CARE_count | "
            "WSM_count | total_count",
            "------|-------|------|------|----------|-------------|--------------|"
            "----------------|----------------|--------------|------------|-----------|------------",
        ]
        for record in self._rows:
            for split, prefix, loss_prefix, care_prefix, wsm_prefix in self._table_splits():
                values = record["logs"]
                for task in _TASKS:
                    loss = values.get(f"{loss_prefix}/loss") if task == "depression" else None
                    care_count = self._count(values, care_prefix, task)
                    wsm_count = self._count(values, wsm_prefix, task)
                    total_count = care_count + wsm_count
                    row = [
                        str(record["epoch"]),
                        split,
                        task,
                        _format(loss),
                    ]
                    row.extend(
                        _format(values.get(f"{prefix}/{task}/{metric}"))
                        for metric in _METRICS
                    )
                    row.extend(
                        [
                            _format(care_count),
                            _format(wsm_count),
                            _format(total_count),
                        ]
                    )
                    lines.append(" | ".join(row))

        self._path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _count(values: dict[str, Any], prefix: str | None, task: str) -> float:
        if prefix is None:
            return 0.0
        return float(values.get(f"{prefix}/{task}/observed_count") or 0.0)

    @staticmethod
    def _table_splits() -> list[tuple[str, str, str, str | None, str | None]]:
        return [
            ("TRAIN", "train/joint", "train", "train/care", "train/wsm"),
            ("VAL CARE", "CARE/care", "CARE", "CARE/care", None),
            ("VAL WSM", "WSM/wsm", "WSM", None, "WSM/wsm"),
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
        experiment = config.section("experiment_info").get("params", {}) if config else {}
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


@CALLBACKS.register("carewsm_summary_callback")
def carewsm_summary_callback(
    context: Any | None = None,
    filename: str = "summary.txt",
    metrics_filename: str = "metrics_history.json",
    **params: Any,
) -> CarewsmSummaryCallback:
    del params
    return CarewsmSummaryCallback(
        context=context,
        filename=filename,
        metrics_filename=metrics_filename,
    )
