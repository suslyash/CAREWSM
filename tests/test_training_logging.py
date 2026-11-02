import sys
from types import SimpleNamespace

sys.path.insert(0, "src")

import torch

from audio.callbacks.summary import CarewsmSummaryCallback
from audio.metrics.classification import MultitaskPRFMetric


_METRICS = (
    "macro_f1",
    "weighted_f1",
    "macro_recall",
    "weighted_recall",
    "patient_recall",
    "healthy_recall",
)


def _metric_block(prefix, count, value):
    result = {f"{prefix}/observed_count": float(count)}
    result.update({f"{prefix}/{name}": value for name in _METRICS})
    return result


def test_test_average_only_averages_metrics_not_observed_counts():
    results = {}
    results.update(_metric_block("test/CARE/care/depression", 10, 0.2))
    results.update(_metric_block("test/WSM/wsm/depression", 2, 0.6))
    results.update(_metric_block("test/CARE/care/parkinson", 4, 0.3))
    results.update(_metric_block("test/WSM/wsm/parkinson", 8, 0.7))
    results["test/CARE/loss"] = 1.0
    results["test/WSM/loss"] = 3.0

    CarewsmSummaryCallback._add_test_averages(results)

    assert results["test/average/depression/macro_f1"] == 0.4
    assert results["test/average/parkinson/macro_f1"] == 0.5
    assert results["test/average/loss"] == 2.0
    assert "test/average/depression/observed_count" not in results
    assert results["test/CARE/care/depression/observed_count"] == 10.0
    assert results["test/WSM/wsm/depression/observed_count"] == 2.0


class _Logger:
    def __init__(self):
        self.log_path = "/tmp/test-training-logging/train.log"
        self.messages = []

    def info(self, *args, **kwargs):
        self.messages.append(("info", args, kwargs))

    def warning(self, *args, **kwargs):
        self.messages.append(("warning", args, kwargs))

    def error(self, *args, **kwargs):
        self.messages.append(("error", args, kwargs))

    def exception(self, *args, **kwargs):
        self.messages.append(("exception", args, kwargs))


def test_per_epoch_test_restores_logger_and_does_not_touch_val_callbacks():
    logger = _Logger()
    selection = SimpleNamespace(
        monitor="joint/average_macro_f1", _saved=["epoch=1.ckpt"], best=0.9
    )
    early = SimpleNamespace(
        monitor="joint/average_macro_f1", best=0.9, bad_epochs=0, stop_training=False
    )
    mlflow_calls = []
    trainer = SimpleNamespace(
        config=SimpleNamespace(device="cpu"),
        logger=logger,
        mlflow_logger=SimpleNamespace(
            log_metrics=lambda values, step: mlflow_calls.append(values)
        ),
        callbacks=[selection, early],
        train_val_state={"marker": [1, 2, 3]},
        _test_loaders={"CARE": object(), "WSM": object()},
    )
    calls = []

    def run_epoch(**kwargs):
        assert trainer.logger is not logger
        assert kwargs["train"] is False
        calls.append(kwargs["split"])
        prefix = "care/depression" if kwargs["split"] == "test_CARE" else "wsm/depression"
        other = "care/parkinson" if kwargs["split"] == "test_CARE" else "wsm/parkinson"
        result = {"loss": 1.0}
        result.update(_metric_block(prefix, 10 if prefix.startswith("care") else 2, 0.2))
        result.update(_metric_block(other, 4 if other.startswith("care") else 8, 0.3))
        return result

    trainer._run_epoch = run_epoch
    callback = CarewsmSummaryCallback()
    callback._test_loaders = trainer._test_loaders
    state_before = dict(trainer.train_val_state)
    selection_before = dict(vars(selection))
    early_before = dict(vars(early))

    result = callback._evaluate_test(trainer, epoch=1)

    assert calls == ["test_CARE", "test_WSM"]
    assert trainer.logger is logger
    assert trainer.train_val_state == state_before
    assert vars(selection) == selection_before
    assert vars(early) == early_before
    assert result["test/average/depression/macro_f1"] == 0.2
    assert "test/average/depression/observed_count" not in result
    assert mlflow_calls
    assert not any(
        key.endswith("/observed_count") and "/average/" in key
        for key in mlflow_calls[0]
    )


class _MetricBatch:
    def __init__(self, targets, observed, corpora):
        self.targets = targets
        self.meta = {"sample_meta": [{"corpus": corpus} for corpus in corpora]}
        self._observed = observed

    def get_masks(self, name):
        assert name == "observed_targets"
        return self._observed


def test_multitask_metric_has_real_counts_and_no_average_support():
    metric = MultitaskPRFMetric()
    targets = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 0]])
    predictions = torch.tensor([[2.0, -2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]])
    observed = torch.tensor([[1, 1], [1, 0], [1, 1], [0, 0]])
    batch = _MetricBatch(targets, observed, ["CARE", "CARE", "WSM", "WSM"])
    metric.update(SimpleNamespace(preds=predictions), batch)

    result = metric.compute()

    assert result["care/depression/observed_count"] == 2.0
    assert result["wsm/depression/observed_count"] == 1.0
    assert result["care/parkinson/observed_count"] == 1.0
    assert result["wsm/parkinson/observed_count"] == 1.0
    assert result["average/depression/macro_f1"] == 0.5
    assert "average/depression/observed_count" not in result
    assert "average/parkinson/observed_count" not in result
    assert not any(key.startswith("joint/") for key in result)


def test_multitask_metric_handles_missing_corpus_and_task_labels():
    metric = MultitaskPRFMetric()
    targets = torch.tensor([[1, 0], [0, 1]])
    predictions = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
    observed = torch.tensor([[1, 0], [1, 0]])
    metric.update(
        SimpleNamespace(preds=predictions),
        _MetricBatch(targets, observed, ["CARE", "CARE"]),
    )

    result = metric.compute()

    assert result["care/depression/observed_count"] == 2.0
    assert result["care/parkinson/observed_count"] == 0.0
    assert result["wsm/depression/observed_count"] == 0.0
    assert result["wsm/parkinson/observed_count"] == 0.0
    assert result["average_macro_f1"] == float("-inf")
    assert not any(key.endswith("/observed_count") and "average" in key for key in result)


def test_best_epoch_is_selected_from_val_monitor_only():
    callback = CarewsmSummaryCallback()
    callback._rows = [
        {
            "epoch": 1,
            "checkpoint": "epoch=1.pt",
            "logs": {
                "joint/average_macro_f1": 0.8,
                "test/average/depression/macro_f1": 0.2,
            },
        },
        {
            "epoch": 2,
            "checkpoint": "epoch=2.pt",
            "logs": {
                "joint/average_macro_f1": 0.7,
                "test/average/depression/macro_f1": 0.9,
            },
        },
    ]

    best = callback._best_row()

    assert best is not None
    assert best["epoch"] == 1
    assert best["checkpoint"] == "epoch=1.pt"
