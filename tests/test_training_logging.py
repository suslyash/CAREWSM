"""Regression tests for the active corpus-specific metric contract."""

from types import SimpleNamespace

import pytest
import torch

from audio.metrics.carewsm_audio_metrics import CAREWSMAudioClassificationMetric
from chimera_ml.data.masking_collate import MaskingCollate


def _batch(targets, observed):
    samples = [
        {
            "inputs": {"native": torch.ones(1, 2)},
            "targets": torch.tensor(target),
            "meta": {"observed_targets": torch.tensor(mask)},
        }
        for target, mask in zip(targets, observed)
    ]
    return MaskingCollate()(samples)


def test_active_metric_reports_six_metrics_and_masked_counts():
    metric = CAREWSMAudioClassificationMetric()
    metric.update(
        SimpleNamespace(
            preds=torch.tensor([[2, 2], [2, -2], [-2, -2], [2, 2]], dtype=torch.float32)
        ),
        _batch(
            [[1, 1], [1, 0], [0, 1], [0, 0]],
            [[True, True], [True, False], [True, True], [True, True]],
        ),
    )
    result = metric.compute()
    for task in ("depression", "parkinson"):
        for name in (
            "macro_f1", "weighted_f1", "macro_recall",
            "weighted_recall", "patient_recall", "healthy_recall",
        ):
            assert f"{task}/{name}" in result


def test_active_metric_marks_single_class_task_unavailable():
    metric = CAREWSMAudioClassificationMetric()
    metric.update(
        SimpleNamespace(preds=torch.tensor([[1.0, 1.0], [1.0, -1.0]])),
        _batch([[1, 0], [1, 0]], [[False, True], [False, True]]),
    )
    result = metric.compute()
    assert torch.isnan(torch.tensor(result["parkinson/macro_recall"]))
