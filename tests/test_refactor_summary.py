"""Focused tests for direct Chimera epoch-log summary persistence."""

import csv
from pathlib import Path
from types import SimpleNamespace

import chimera_plugin
from audio.callbacks.carewsm_summary_callback import CAREWSMSummaryCallback
from chimera_ml.core.registry import CALLBACKS


def test_summary_writes_one_dynamic_wide_row_and_is_idempotent(tmp_path: Path) -> None:
    logger = SimpleNamespace(log_path=str(tmp_path / "logs" / "train.log"))
    callback = CAREWSMSummaryCallback()
    callback.on_fit_start(SimpleNamespace(logger=logger))
    logs = {
        "train/num_samples": 4,
        "val_care/depression/macro_recall": 0.5,
        "val_wsm/depression/macro_recall": 0.75,
        "test_care/depression/macro_recall": 0.25,
        "test_wsm/depression/macro_recall": 0.125,
        "dev_mean_macro_recall": 0.625,
    }
    callback.on_epoch_end(SimpleNamespace(), 1, logs)
    callback.on_epoch_end(SimpleNamespace(), 1, logs)
    path = tmp_path / "logs" / "summary.txt"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8"), delimiter=";"))
    assert len(rows) == 1
    assert rows[0]["val_care_depression_macro_recall"] == "0.5000"
    assert rows[0]["test_wsm_depression_macro_recall"] == "0.1250"
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_summary_factory_keeps_registry_identifier_and_class_name() -> None:
    chimera_plugin.register()
    callback = CALLBACKS._items["carewsm_summary_callback"]()
    assert isinstance(callback, CAREWSMSummaryCallback)
