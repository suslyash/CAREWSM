"""Focused tests for the corpus-mode wide summary and concise epoch logging."""

import csv
import io
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from audio.callbacks.summary import CAREWSMSummaryCallback
import chimera_plugin
from chimera_ml.core.registry import CALLBACKS


class _Logger:
    def __init__(self, path: Path) -> None:
        self.log_path = str(path / "train.log")
        self.messages: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def warning(self, message: str) -> None:
        self.messages.append(("warning", message))

    def error(self, message: str) -> None:
        self.messages.append(("error", message))


class _Context:
    def __init__(self, data_module) -> None:
        self.data_module = data_module

    def get(self, name: str):
        assert name == "data.module"
        return self.data_module


def _logs() -> dict[str, float]:
    logs: dict[str, float] = {"dev_mean_macro_recall": 0.625}
    for population in ("train", "val_care", "val_wsm", "test_care", "test_wsm"):
        logs[f"{population}/loss"] = 0.5
        logs[f"{population}/num_samples"] = 4
        for task in ("depression", "parkinson"):
            prefix = f"{population}/{task}"
            logs[f"{prefix}/observed_count"] = 3
            logs[f"{prefix}/positive_support"] = 2
            logs[f"{prefix}/negative_support"] = 1
            for metric in (
                "macro_f1",
                "weighted_f1",
                "macro_recall",
                "weighted_recall",
                "patient_recall",
                "healthy_recall",
            ):
                logs[f"{prefix}/{metric}"] = 0.25
    return logs


def _test_metrics(value: float) -> dict[str, float]:
    result = {"loss": value, "num_samples": 2}
    for task in ("depression", "parkinson"):
        prefix = f"{task}/"
        result[f"{prefix}observed_count"] = 2
        result[f"{prefix}positive_support"] = 1
        result[f"{prefix}negative_support"] = 1
        for metric in (
            "macro_f1",
            "weighted_f1",
            "macro_recall",
            "weighted_recall",
            "patient_recall",
            "healthy_recall",
        ):
            result[f"{prefix}{metric}"] = value
    return result


def test_corpus_summary_evaluates_real_test_flow_once_per_epoch(tmp_path):
    logger = _Logger(tmp_path / "logs" / "experiment" / "run")
    data_module = SimpleNamespace(
        evaluation_layout="corpus",
        test_dataloader=lambda: {"test_care": object(), "test_wsm": object()},
    )
    callback = CAREWSMSummaryCallback(context=_Context(data_module))
    trainer = SimpleNamespace(
        logger=logger,
        callbacks=[],
        config=SimpleNamespace(device="cpu"),
        mlflow_logger=None,
    )
    calls: list[str] = []

    def run_epoch(**kwargs):
        split = kwargs["split"]
        calls.append(split)
        logger.info(f"[{split} epoch 1] native")
        return _test_metrics(0.91 if split == "test_care" else 0.83)

    trainer._run_epoch = run_epoch
    callback.on_fit_start(trainer)
    logs = {
        key: value
        for key, value in _logs().items()
        if not key.startswith("test_")
    }
    logger.info("[train epoch 1] native")
    logger.info("[val_care epoch 1] native")
    logger.info("[val_wsm epoch 1] native")
    callback.on_epoch_end(trainer, 1, logs)
    assert calls == ["test_care", "test_wsm"]
    first_messages = list(logger.messages)

    callback.on_epoch_end(trainer, 1, logs)
    assert calls == ["test_care", "test_wsm"]
    assert logger.messages == first_messages
    assert len([message for level, message in logger.messages if level == "info"]) == 5

    summary = Path(logger.log_path).parent / "summary.txt"
    rows = list(
        csv.DictReader(
            summary.open(newline="", encoding="utf-8"),
            delimiter=";",
        )
    )
    assert len(rows) == 1
    assert rows[0]["test_care_depression_macro_f1"] == "0.9100"
    assert rows[0]["test_wsm_parkinson_healthy_recall"] == "0.8300"
    assert "test/average/loss" not in callback._corpus_test_results[1]
    assert [message.split("]")[0][1:].split()[0] for level, message in logger.messages if level == "info"] == [
        "train",
        "val_care",
        "val_wsm",
        "test_care",
        "test_wsm",
    ]



def test_corpus_test_mapping_uses_original_logger_and_corpus_keys(tmp_path):
    original_logger = _Logger(tmp_path)
    callback = CAREWSMSummaryCallback()
    callback._corpus_mode = True
    callback._test_loaders = {"test_care": object(), "test_wsm": object()}
    trainer = SimpleNamespace(
        logger=original_logger,
        config=SimpleNamespace(device="cpu"),
        mlflow_logger=None,
    )

    calls: list[str] = []

    def run_epoch(**kwargs):
        calls.append(kwargs["split"])
        return {"loss": 0.2, "num_samples": 2}

    trainer._run_epoch = run_epoch
    results = callback._evaluate_test(trainer, 1)

    assert calls == ["test_care", "test_wsm"]
    assert trainer.logger is original_logger
    assert results["test_care/loss"] == 0.2
    assert results["test_wsm/num_samples"] == 2


def _real_logger(tmp_path, name: str):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.filters.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    logger.log_path = str(tmp_path / "train.log")
    return logger, handler, stream


def test_test_failure_leaves_logger_and_handlers_unchanged(tmp_path):
    logger, handler, stream = _real_logger(tmp_path, "carewsm-summary-failure")
    data_module = SimpleNamespace(
        evaluation_layout="corpus",
        test_dataloader=lambda: {"test_care": object()},
    )
    callback = CAREWSMSummaryCallback(context=_Context(data_module))
    trainer = SimpleNamespace(
        logger=logger,
        config=SimpleNamespace(device="cpu"),
        mlflow_logger=None,
    )
    original_logger = trainer.logger
    callback.on_fit_start(trainer)

    def failing_epoch(**kwargs):
        logger.warning("test warning")
        logger.error("test error")
        raise RuntimeError("synthetic TEST failure")

    trainer._run_epoch = failing_epoch
    with pytest.raises(RuntimeError, match="synthetic TEST failure"):
        callback.on_epoch_end(trainer, 1, _logs())
    assert trainer.logger is original_logger
    assert not handler.filters
    assert "test warning" in stream.getvalue()
    assert "test error" in stream.getvalue()
    logger.removeHandler(handler)
    handler.close()


def test_summary_factory_keeps_registry_name_and_new_class_name():
    chimera_plugin.register()
    callback = CALLBACKS._items["carewsm_summary_callback"]()
    assert isinstance(callback, CAREWSMSummaryCallback)
    assert callback.__class__.__name__ == "CAREWSMSummaryCallback"
