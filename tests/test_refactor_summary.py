"""Focused tests for the corpus-mode wide summary and concise epoch logging."""

import csv
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from audio.callbacks.summary import CAREWSMSummaryCallback, _TrainerEpochFilter
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


def test_corpus_summary_has_one_wide_semicolon_row_and_five_info_lines(tmp_path):
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
    )
    callback.on_fit_start(trainer)
    callback._test_loaders = {}
    callback.on_epoch_end(trainer, 1, _logs())
    logger.messages.clear()
    callback.on_epoch_end(trainer, 1, _logs())

    summary = Path(logger.log_path).parent / "summary.txt"
    rows = list(
        csv.DictReader(
            summary.open(newline="", encoding="utf-8"),
            delimiter=";",
        )
    )
    assert len(rows) == 1
    assert rows[0]["epoch"] == "1"
    assert rows[0]["val_care_depression_macro_recall"] == "0.2500"
    assert rows[0]["test_wsm_parkinson_healthy_recall"] == "0.2500"
    assert rows[0]["dev_mean_macro_recall"] == "0.6250"
    assert "val_care_depression_macro_f1" in rows[0]
    assert "val_care_parkinson_weighted_recall" in rows[0]
    assert summary.read_text(encoding="utf-8").count("\n") == 2

    messages = [message for level, message in logger.messages if level == "info"]
    assert len(messages) == 5
    assert [message.split("split=")[1].split()[0] for message in messages] == [
        "train",
        "val_care",
        "val_wsm",
        "test_care",
        "test_wsm",
    ]
    assert all("macro_f1" not in message for message in messages)
    callback.on_fit_end(trainer)


def test_corpus_test_mapping_restores_logger_and_uses_corpus_keys(tmp_path):
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


def test_trainer_epoch_filter_suppresses_only_new_stock_info_lines():
    epoch_filter = _TrainerEpochFilter()
    assert not epoch_filter.filter(
        logging.LogRecord(
            "chimera", logging.INFO, "", 0, "[val_care epoch 1] verbose", (), None
        )
    )
    assert epoch_filter.filter(
        logging.LogRecord(
            "chimera", logging.INFO, "", 0, "[epoch summary] concise", (), None
        )
    )
    assert epoch_filter.filter(
        logging.LogRecord(
            "chimera", logging.WARNING, "", 0, "[val_care epoch 1] warning", (), None
        )
    )


def test_summary_factory_keeps_registry_name_and_new_class_name():
    chimera_plugin.register()
    callback = CALLBACKS._items["carewsm_summary_callback"]()
    assert isinstance(callback, CAREWSMSummaryCallback)
    assert callback.__class__.__name__ == "CAREWSMSummaryCallback"
