import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from audio.callbacks.summary import CAREWSMSummaryCallback
from audio.callbacks.validation_monitor import CAREWSMValidationMonitorCallback
from audio.loss.masked_bce import CAREWSMMaskedBCELoss
from audio.metrics.classification import CAREWSMClassificationMetric
from audio.models.classifiers import PooledMLP
from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.callbacks.checkpoint_callback import CheckpointCallback
from chimera_ml.callbacks.early_stopping_callback import EarlyStoppingCallback
from chimera_ml.callbacks.snapshot_callback import SnapshotCallback
from chimera_ml.logging.console_file_logger import ConsoleFileLogger
from chimera_ml.data.masking_collate import MaskingCollate
from chimera_ml.training.config import TrainConfig
from chimera_ml.training.trainer import Trainer


class _RecordingMLP(PooledMLP):
    def __init__(self):
        super().__init__(input_dim=2, hidden_dim=4, dropout=0.0)
        self.train_seen_corpora = []

    def forward(self, batch):
        if self.training:
            self.train_seen_corpora.extend(
                item["corpus"] for item in batch.meta["sample_meta"]
            )
        return super().forward(batch)


class _CaptureLogs(BaseCallback):
    def __init__(self):
        self.logs = None

    def on_epoch_end(self, trainer, epoch, logs):
        del trainer, epoch
        self.logs = dict(logs)


class _Context:
    def __init__(self, data_module):
        self.data_module = data_module

    def get(self, name):
        assert name == "data.module"
        return self.data_module


def _samples(corpus, count=4, flip=False):
    samples = []
    for index in range(count):
        target = torch.tensor([index % 2, (index + 1) % 2], dtype=torch.float32)
        if flip:
            target = 1.0 - target
        samples.append(
            {
                "inputs": {"native": torch.tensor([[0.0, 0.0]] if flip and index % 2 else [[-1.0, 1.0]] if flip else [[1.0, -1.0]])},
                "targets": target,
                "meta": {
                    "corpus": corpus,
                    "observed_targets": torch.tensor([True, True]),
                },
            }
        )
    return samples


def _loader_from_samples(samples):
    return DataLoader(
        samples,
        batch_size=2,
        shuffle=False,
        collate_fn=MaskingCollate(),
    )


def _loader(corpus, count=4, flip=False):
    return DataLoader(
        _samples(corpus, count, flip=flip),
        batch_size=2,
        shuffle=False,
        collate_fn=MaskingCollate(),
    )


def test_canonical_yaml_inventory_is_110_and_unique():
    paths = sorted(
        path
        for directory in Path("configs/audio").iterdir()
        if directory.is_dir()
        for path in directory.glob("*_fold[0-4].yaml")
    )
    identities = {(path.parent.name, path.stem) for path in paths}
    assert len(paths) == 110
    assert len(identities) == 110


def test_real_trainer_fit_uses_joint_train_and_native_logging(tmp_path):
    run_root = tmp_path / "logs"
    logger = ConsoleFileLogger(
        log_path=run_root,
        experiment_name="synthetic_audio",
        run_name="pooled_mlp_fold0",
        name="carewsm-synthetic-fit",
        console_level="INFO",
        file_level="INFO",
    )
    data_module = SimpleNamespace(
        evaluation_layout="corpus",
        test_dataloader=lambda: {
            "test_care": _loader("CARE", flip=True),
            "test_wsm": _loader("WSM", flip=True),
        },
    )
    summary = CAREWSMSummaryCallback(context=_Context(data_module))
    checkpoint = CheckpointCallback(
        log_path=str(run_root),
        experiment_name="synthetic_audio",
        run_name="pooled_mlp_fold0",
        monitor="dev_mean_macro_recall",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    early_stopping = EarlyStoppingCallback(
        monitor="dev_mean_macro_recall",
        mode="max",
        patience=2,
    )
    capture = _CaptureLogs()
    callbacks = [
        CAREWSMValidationMonitorCallback(),
        checkpoint,
        early_stopping,
        summary,
        SnapshotCallback(
            log_path=str(run_root),
            experiment_name="synthetic_audio",
            run_name="pooled_mlp_fold0",
            save_code_zip=False,
            save_config=False,
        ),
        capture,
    ]
    assert [callback.__class__.__name__ for callback in callbacks[:4]] == [
        "CAREWSMValidationMonitorCallback",
        "CheckpointCallback",
        "EarlyStoppingCallback",
        "CAREWSMSummaryCallback",
    ]

    torch.manual_seed(0)
    model = _RecordingMLP()
    trainer = Trainer(
        model=model,
        loss_fn=CAREWSMMaskedBCELoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        metrics=[CAREWSMClassificationMetric()],
        config=TrainConfig(epochs=1, device="cpu", mixed_precision=False),
        logger=logger,
        callbacks=callbacks,
    )
    train_samples = _samples("CARE", 2) + _samples("WSM", 2)
    train_samples[0]["meta"]["observed_targets"] = torch.tensor([True, False])
    trainer.fit(
        _loader_from_samples(train_samples),
        val_loaders={"val_care": _loader("CARE"), "val_wsm": _loader("WSM")},
    )

    run_dir = run_root / "synthetic_audio" / "pooled_mlp_fold0"
    log_text = (run_dir / "train.log").read_text(encoding="utf-8")
    epoch_lines = [
        line
        for line in log_text.splitlines()
        if any(f"[{split} epoch 1]" in line for split in ("train", "val_care", "val_wsm", "test_care", "test_wsm"))
    ]
    assert len(epoch_lines) == 5
    assert [next(split for split in ("train", "val_care", "val_wsm", "test_care", "test_wsm") if f"[{split} epoch 1]" in line) for line in epoch_lines] == ["train", "val_care", "val_wsm", "test_care", "test_wsm"]
    assert "[epoch summary]" not in log_text
    assert model.train_seen_corpora.count("CARE") == 2
    assert model.train_seen_corpora.count("WSM") == 2
    assert capture.logs is not None
    assert capture.logs["train/num_samples"] == 4
    assert log_text.index("[val_wsm epoch 1]") < log_text.index("[test_care epoch 1]")

    validation_values = [capture.logs[name] for name in CAREWSMValidationMonitorCallback.components]
    assert capture.logs["dev_mean_macro_recall"] == sum(validation_values) / 4
    test_recall_pairs = [
        (
            capture.logs[f"val_{corpus}/depression/macro_recall"],
            summary._corpus_test_results[1][f"test_{corpus}/depression/macro_recall"],
        )
        for corpus in ("care", "wsm")
    ]
    test_recall_pairs.extend(
        (
            capture.logs[f"val_{corpus}/parkinson/macro_recall"],
            summary._corpus_test_results[1][f"test_{corpus}/parkinson/macro_recall"],
        )
        for corpus in ("care", "wsm")
    )
    assert any(validation != test for validation, test in test_recall_pairs)
    assert summary._corpus_test_results[1]["test_care/loss"] >= 0.0
    assert summary._corpus_test_results[1]["test_wsm/loss"] >= 0.0
    assert checkpoint._best == capture.logs["dev_mean_macro_recall"]
    assert early_stopping._best == capture.logs["dev_mean_macro_recall"]
    assert checkpoint._saved
    checkpoint_name = checkpoint._saved[0].name
    assert "dev_mean_macro_recall" in checkpoint_name
    assert "test_" not in checkpoint_name

    summary_path = run_dir / "summary.txt"
    rows = summary_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    header = rows[0].split(";")
    assert len(rows[1].split(";")) == len(header)
    for population in ("train", "val_care", "val_wsm", "test_care", "test_wsm"):
        for task in ("depression", "parkinson"):
            for metric in ("macro_f1", "weighted_f1", "macro_recall", "weighted_recall", "patient_recall", "healthy_recall"):
                assert f"{population}_{task}_{metric}" in header
    assert (run_dir / "checkpoints").is_dir()
    assert list((run_dir / "checkpoints").glob("*.pt"))


def test_missing_canonical_yaml_fails_full_matrix_check(tmp_path):
    source = Path("configs/audio")
    fixture = tmp_path / "configs" / "audio"
    shutil.copytree(source, fixture)
    missing = fixture / "emotionmsp_embeds" / "mlp_fold0.yaml"
    missing.unlink()
    script = Path("scripts/run_audio_experiments.sh").resolve()
    result = subprocess.run(
        ["bash", str(script), "--check", "--all"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Expected 110 canonical files" in result.stderr


def test_missing_validation_component_fails_before_checkpoint(tmp_path):
    model = PooledMLP(input_dim=2, hidden_dim=4, dropout=0.0)
    trainer = SimpleNamespace(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        logger=None,
        scheduler=None,
        global_step=0,
        stop_training=False,
    )
    monitor = CAREWSMValidationMonitorCallback()
    checkpoint = CheckpointCallback(
        log_path=str(tmp_path),
        experiment_name="invalid",
        run_name="missing-val",
        monitor="dev_mean_macro_recall",
        mode="max",
        save_last=False,
    )
    checkpoint.on_fit_start(trainer)
    logs = {
        component: 0.5
        for component in CAREWSMValidationMonitorCallback.components[:-1]
    }
    try:
        monitor.on_epoch_end(trainer, 1, logs)
    except ValueError as exc:
        assert "unavailable" in str(exc)
    else:
        raise AssertionError("missing validation component was accepted")
    assert not checkpoint._saved
    assert not list((tmp_path / "invalid" / "missing-val" / "checkpoints").glob("*.pt"))
