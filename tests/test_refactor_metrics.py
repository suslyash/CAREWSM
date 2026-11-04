"""Focused tests for corpus-specific masked metrics and validation selection."""

from types import SimpleNamespace

import pytest
import torch

from audio.callbacks.validation_monitor import CAREWSMValidationMonitorCallback
from chimera_ml.callbacks.checkpoint_callback import CheckpointCallback
from chimera_ml.callbacks.early_stopping_callback import EarlyStoppingCallback
from audio.metrics.classification import CAREWSMClassificationMetric
from chimera_ml.data.masking_collate import MaskingCollate


def _batch(targets: list[list[float]], observed: list[list[bool]]):
    samples = [
        {
            "inputs": {"native": torch.ones(1, 2)},
            "targets": torch.tensor(target),
            "meta": {"observed_targets": torch.tensor(mask)},
        }
        for target, mask in zip(targets, observed)
    ]
    return MaskingCollate()(samples)


def test_masked_metric_uses_collated_masks_and_all_six_formulas():
    metric = CAREWSMClassificationMetric()
    batch = _batch(
        [[1, 1], [1, 0], [1, 1], [0, 0]],
        [[True, True], [True, False], [True, True], [True, True]],
    )
    output = SimpleNamespace(
        preds=torch.tensor([[2, 2], [2, -2], [2, -2], [2, 2]], dtype=torch.float32)
    )
    metric.update(output, batch)
    result = metric.compute()

    assert result["depression/observed_count"] == 4.0
    assert result["depression/positive_support"] == 3.0
    assert result["depression/negative_support"] == 1.0
    assert result["depression/patient_recall"] == 1.0
    assert result["depression/healthy_recall"] == 0.0
    assert result["depression/macro_recall"] == 0.5
    assert result["depression/weighted_recall"] == 0.75
    assert result["depression/macro_f1"] == pytest.approx(3 / 7)
    assert result["depression/weighted_f1"] == pytest.approx(9 / 14)
    assert result["parkinson/observed_count"] == 3.0
    assert result["parkinson/positive_support"] == 2.0
    assert result["parkinson/negative_support"] == 1.0


def test_masked_metric_marks_empty_and_single_class_tasks_unavailable():
    metric = CAREWSMClassificationMetric()
    metric.update(
        SimpleNamespace(preds=torch.tensor([[1.0, 1.0], [1.0, -1.0]])),
        _batch([[1, 0], [1, 0]], [[False, True], [False, True]]),
    )
    result = metric.compute()

    assert result["depression/observed_count"] == 0.0
    assert torch.isnan(torch.tensor(result["depression/macro_recall"]))
    assert result["parkinson/observed_count"] == 2.0
    assert result["parkinson/positive_support"] == 0.0
    assert result["parkinson/negative_support"] == 2.0
    assert torch.isnan(torch.tensor(result["parkinson/macro_f1"]))


def _valid_logs():
    return {
        name: value
        for name, value in zip(
            CAREWSMValidationMonitorCallback.components, (0.2, 0.4, 0.6, 0.8)
        )
    }


def test_validation_monitor_is_equal_weight_and_validation_only():
    logs = _valid_logs()
    CAREWSMValidationMonitorCallback().on_epoch_end(SimpleNamespace(logger=None), 1, logs)
    assert logs["dev_mean_macro_recall"] == 0.5
    assert "test_care/depression/macro_recall" not in logs


def test_validation_monitor_requires_all_finite_components():
    logs = _valid_logs()
    del logs[CAREWSMValidationMonitorCallback.components[-1]]
    with pytest.raises(ValueError, match="unavailable"):
        CAREWSMValidationMonitorCallback().on_epoch_end(SimpleNamespace(logger=None), 1, logs)


class _CallbackLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def warning(self, message: str) -> None:
        self.messages.append(("warning", message))


def _selection_trainer(tmp_path, logger):
    model = torch.nn.Linear(1, 1)
    return SimpleNamespace(
        logger=logger,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=None,
        global_step=3,
        stop_training=False,
    )

def test_real_callbacks_consume_monitor_in_trainer_order(tmp_path):
    logger = _CallbackLogger()
    trainer = _selection_trainer(tmp_path, logger)
    monitor = CAREWSMValidationMonitorCallback()
    checkpoint = CheckpointCallback(
        log_path=str(tmp_path),
        experiment_name="integration",
        run_name="selection",
        monitor="dev_mean_macro_recall",
        mode="max",
        save_last=False,
    )
    early_stopping = EarlyStoppingCallback(
        monitor="dev_mean_macro_recall", mode="max", patience=2
    )
    callbacks = [monitor, checkpoint, early_stopping]
    callback_names = [callback.__class__.__name__ for callback in callbacks]
    assert callback_names == [
        "CAREWSMValidationMonitorCallback",
        "CheckpointCallback",
        "EarlyStoppingCallback",
    ]
    checkpoint.on_fit_start(trainer)

    logs = _valid_logs()
    logs["test_care/depression/macro_recall"] = 0.99
    for callback in callbacks:
        callback.on_epoch_end(trainer, 1, logs)

    assert logs["dev_mean_macro_recall"] == 0.5
    assert checkpoint._best == logs["dev_mean_macro_recall"]
    assert len(checkpoint._saved) == 1
    assert early_stopping._best == logs["dev_mean_macro_recall"]
    assert not trainer.stop_training
    assert checkpoint._saved[0].exists()

def test_invalid_monitor_fails_before_real_checkpoint_selection(tmp_path):
    logger = _CallbackLogger()
    trainer = _selection_trainer(tmp_path, logger)
    monitor = CAREWSMValidationMonitorCallback()
    checkpoint = CheckpointCallback(
        log_path=str(tmp_path),
        experiment_name="integration",
        run_name="invalid",
        monitor="dev_mean_macro_recall",
        mode="max",
        save_last=False,
    )
    checkpoint.on_fit_start(trainer)
    logs = _valid_logs()
    del logs[CAREWSMValidationMonitorCallback.components[-1]]

    with pytest.raises(ValueError, match="unavailable"):
        monitor.on_epoch_end(trainer, 1, logs)
    assert checkpoint._saved == []
    assert not list((tmp_path / "integration" / "invalid" / "checkpoints").glob("*.pt"))
