"""AV-007 regression checks for automatic fusion VAL and TEST evaluation."""

from pathlib import Path

import yaml


def test_fusion_training_exposes_audio_style_four_evaluation_loaders():
    source = Path("src/fusion/data/carewsm_trimodal_datamodule.py").read_text()
    assert '"val_care": datasets["val_care"]' in source
    assert '"val_wsm": datasets["val_wsm"]' in source
    assert '"test_care": datasets["test_care"]' in source
    assert '"test_wsm": datasets["test_wsm"]' in source
    assert "test_dataset=None" in source


def test_all_fusion_configs_keep_validation_only_selection_monitor():
    paths = list(Path("configs/fusion").glob("*.yaml")) + list(
        Path("configs/fusion/sweep").glob("*.yaml")
    )
    assert len(paths) == 300
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        monitors = [
            callback.get("params", {}).get("monitor")
            for callback in config.get("callbacks", [])
        ]
        assert monitors.count("val_mean_corpora_tasks/macro_recall") == 2
        assert all("test_" not in (monitor or "") for monitor in monitors)


def test_summary_header_has_all_48_corpus_task_split_metrics(tmp_path):
    from audio.callbacks.carewsm_summary_callback import CAREWSMSummaryCallback

    callback = CAREWSMSummaryCallback()

    class Logger:
        log_path = str(tmp_path / "run" / "train.log")

    class Trainer:
        logger = Logger()

    callback.on_fit_start(Trainer())
    logs = {}
    for split in ("val", "test"):
        for corpus in ("care", "wsm"):
            for task in ("depression", "parkinson"):
                for metric in (
                    "macro_f1",
                    "weighted_f1",
                    "macro_recall",
                    "weighted_recall",
                    "patient_recall",
                    "healthy_recall",
                ):
                    logs[f"{split}_{corpus}/{task}/{metric}"] = 0.5
    callback.on_epoch_end(Trainer(), 1, logs)
    header = (tmp_path / "run" / "summary.txt").read_text().splitlines()[0].split(";")
    assert len(
        [field for field in header if field.startswith(("val_", "test_"))]
    ) == 48
