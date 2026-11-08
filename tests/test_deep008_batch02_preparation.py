from __future__ import annotations

import os
import subprocess
from copy import deepcopy
from pathlib import Path

import torch
import yaml

import chimera_plugin
from audio.loss.carewsm_weighted_bce_loss import (
    CAREWSMSingleTaskBCELoss,
    CAREWSMWeightedParkinsonBCELoss,
)
from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS

ROOT = Path("configs/audio_representation_pilot/deep008")
BATCH_ROOT = ROOT / "batch02"
BASELINE_ROOT = ROOT / "weighted_f"
LAUNCHER = Path("scripts/run_audio_deep008_batch02.sh")
EXPERIMENT = "audio_representation_pilot_deep008_batch02"


def _normalise(config: dict) -> dict:
    value = deepcopy(config)
    value["model"]["name"] = "MODEL"
    value["experiment_info"]["params"] = {
        "experiment_name": "EXPERIMENT",
        "run_name": "RUN",
        "include_time": False,
    }
    for item in value["callbacks"]:
        item["params"].pop("task", None)
        if "log_path" in item["params"]:
            item["params"]["log_path"] = "LOG"
    for item in value["logging"]:
        if "log_path" in item["params"]:
            item["params"]["log_path"] = "LOG"
        if "tracking_uri" in item["params"]:
            item["params"]["tracking_uri"] = "TRACKING"
    return value


def _batch(task: str | None = None) -> Batch:
    meta = None
    if task is not None:
        meta = {"sample_meta": [
            {"corpus": "CARE", "observed_targets": torch.tensor([True, True])},
            {"corpus": "WSM", "observed_targets": torch.tensor([True, True])},
        ]}
    return Batch(
        inputs={"native": torch.randn(2, 3, 2048)},
        targets=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        masks={
            "sequence_mask": torch.tensor([[True, True, False], [True, False, False]]),
            "observed_targets": torch.tensor([[True, True], [True, True]]),
        },
        meta=meta,
    )


def test_exactly_fifteen_configs_and_monitor_contract() -> None:
    names = sorted(path.name for path in BATCH_ROOT.glob("*.yaml"))
    assert len(names) == 16
    assert "H2_corpus_heads_fold0_recovery.yaml" in names
    for fold in range(5):
        depression = yaml.safe_load((BATCH_ROOT / f"H1_depression_fold{fold}.yaml").read_text())
        parkinson = yaml.safe_load((BATCH_ROOT / f"H1_parkinson_fold{fold}.yaml").read_text())
        corpus = yaml.safe_load((BATCH_ROOT / f"H2_corpus_heads_fold{fold}.yaml").read_text())
        assert depression["callbacks"][1]["params"]["monitor"] == "val_mean_corpora/depression/macro_recall"
        assert parkinson["callbacks"][1]["params"]["monitor"] == "val_mean_corpora/parkinson/macro_recall"
        assert corpus["callbacks"][1]["params"]["monitor"] == "val_mean_corpora_tasks/macro_recall"
        assert depression["model"]["name"] == "carewsm_gated_single_task"
        assert parkinson["loss"]["params"]["task"] == "parkinson"
        assert corpus["model"]["name"] == "carewsm_gated_corpus_heads"
        assert all(EXPERIMENT in yaml.safe_dump(config) for config in (depression, parkinson, corpus))


def test_h2_is_weighted_f_parity_except_model_identity_and_paths() -> None:
    for fold in range(5):
        baseline = yaml.safe_load((BASELINE_ROOT / f"F_weighted_parkinson_fold{fold}.yaml").read_text())
        candidate = yaml.safe_load((BATCH_ROOT / f"H2_corpus_heads_fold{fold}.yaml").read_text())
        assert _normalise(candidate) == _normalise(baseline)
        assert candidate["model"]["name"] == "carewsm_gated_corpus_heads"


def test_registered_models_and_single_task_masked_backward() -> None:
    chimera_plugin.register()
    batch = _batch()
    for task, index in (("depression", 0), ("parkinson", 1)):
        model = MODELS.create(
            "carewsm_gated_single_task",
            active_task=task,
            input_dim=2048,
            family_dims=[1024, 1024],
            hidden_dim=8,
            dropout=0.0,
        )
        output = model(batch)
        assert output.preds.shape == (2, 2)
        assert torch.all(output.preds[:, 1 - index] == 0)
        loss = CAREWSMSingleTaskBCELoss(
            task, parkinson_pos_weight=2.3 if task == "parkinson" else None
        )(output, batch)
        loss.backward()
        assert torch.isfinite(loss)
        assert model.heads.depression[-1].weight.grad is not None if index == 0 else model.heads.parkinson[-1].weight.grad is not None


def test_corpus_heads_route_metadata_and_launcher_is_cpu_safe() -> None:
    chimera_plugin.register()
    model = MODELS.create(
        "carewsm_gated_corpus_heads",
        input_dim=2048,
        family_dims=[1024, 1024],
        hidden_dim=8,
        dropout=0.0,
    )
    output = model(_batch("corpus"))
    assert output.preds.shape == (2, 2)
    assert all(parameter.grad is None for parameter in model.parameters())
    source = LAUNCHER.read_text()
    assert source.count("$CONFIG_ROOT/H1_") == 10
    assert source.count("$CONFIG_ROOT/H2_") == 10
    assert "$CONFIG_ROOT/H2_corpus_heads_fold0_recovery.yaml" in source
    assert '"--check-h2"' in source
    assert '"--resume-h2"' in source
    assert 'if [[ "$mode" == "--check" || "$mode" == "--check-h2" ]]' in source
    assert "CUDA_VISIBLE_DEVICES=0" in source
    assert "audio_representation_pilot_deep008_batch02" in source
    assert "--all" in source
    assert "TEST" not in source


def test_corpus_heads_float32_and_cpu_bfloat16_autocast_backward() -> None:
    chimera_plugin.register()
    for autocast_enabled in (False, True):
        model = MODELS.create(
            "carewsm_gated_corpus_heads",
            input_dim=2048,
            family_dims=[1024, 1024],
            hidden_dim=8,
            dropout=0.0,
        )
        model.train()
        batch = _batch("corpus")
        with torch.autocast(
            device_type="cpu", dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            output = model(batch)
            loss = CAREWSMWeightedParkinsonBCELoss(2.3)(output, batch)
        loss.backward()
        assert torch.isfinite(output.preds).all()
        assert torch.isfinite(loss)
        assert model.projections[0][1].weight.grad is not None
        for corpus in ("CARE", "WSM"):
            assert model.corpus_heads[corpus].depression[-1].weight.grad is not None
            assert model.corpus_heads[corpus].parkinson[-1].weight.grad is not None
