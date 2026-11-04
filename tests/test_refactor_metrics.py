"""Focused tests for corpus-specific masked metrics and validation selection."""

from types import SimpleNamespace

import pytest
import torch

from audio.callbacks.validation_monitor import CarewsmValidationMonitor
from audio.metrics.classification import MaskedMultitaskClassificationMetric
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
    metric = MaskedMultitaskClassificationMetric()
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
    metric = MaskedMultitaskClassificationMetric()
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
            CarewsmValidationMonitor.components, (0.2, 0.4, 0.6, 0.8)
        )
    }


def test_validation_monitor_is_equal_weight_and_validation_only():
    logs = _valid_logs()
    CarewsmValidationMonitor().on_epoch_end(SimpleNamespace(logger=None), 1, logs)
    assert logs["dev_mean_macro_recall"] == 0.5
    assert "test_care/depression/macro_recall" not in logs


def test_validation_monitor_requires_all_finite_components():
    logs = _valid_logs()
    del logs[CarewsmValidationMonitor.components[-1]]
    with pytest.raises(ValueError, match="unavailable"):
        CarewsmValidationMonitor().on_epoch_end(SimpleNamespace(logger=None), 1, logs)


def test_monitor_must_precede_selection_callbacks():
    order = ["monitor", "checkpoint", "early_stopping"]
    assert order.index("monitor") < order.index("checkpoint")
    assert order.index("monitor") < order.index("early_stopping")
