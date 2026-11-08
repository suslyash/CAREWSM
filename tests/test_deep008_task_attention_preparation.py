from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

import chimera_plugin
from audio.loss.carewsm_weighted_bce_loss import CAREWSMWeightedParkinsonBCELoss
from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS


BASELINE_ROOT = Path("configs/audio_representation_pilot/deep008/weighted_f")
TASK_ROOT = Path("configs/audio_representation_pilot/deep008/task_attention")
LAUNCHER = Path("scripts/run_audio_deep008_task_attention.sh")
EXPERIMENT = "audio_representation_pilot_deep008_task_attention"
LOG_PATH = "logs/audio_representation_pilot/deep008_task_attention"
TRACKING_URI = "sqlite:///logs/audio_representation_pilot/deep008_task_attention/mlflow.db"


def _config_pair(fold: int) -> tuple[dict, dict]:
    baseline_path = BASELINE_ROOT / f"F_weighted_parkinson_fold{fold}.yaml"
    task_path = TASK_ROOT / f"F_task_attention_fold{fold}.yaml"
    return (
        yaml.safe_load(baseline_path.read_text()),
        yaml.safe_load(task_path.read_text()),
    )


def _normalized_for_parity(config: dict) -> dict:
    normalized = deepcopy(config)
    normalized["model"]["name"] = "MODEL_NAME"
    params = normalized["experiment_info"]["params"]
    params["experiment_name"] = "EXPERIMENT_NAME"
    params["run_name"] = "RUN_NAME"
    for callback in normalized["callbacks"]:
        callback_params = callback["params"]
        if "log_path" in callback_params:
            callback_params["log_path"] = "LOG_PATH"
    for logger in normalized["logging"]:
        logger_params = logger["params"]
        if "log_path" in logger_params:
            logger_params["log_path"] = "LOG_PATH"
        if "tracking_uri" in logger_params:
            logger_params["tracking_uri"] = "TRACKING_URI"
    return normalized


def test_five_task_attention_configs_have_only_approved_differences() -> None:
    task_paths = sorted(TASK_ROOT.glob("F_task_attention_fold[0-4].yaml"))
    assert [path.name for path in task_paths] == [
        f"F_task_attention_fold{fold}.yaml" for fold in range(5)
    ]

    for fold in range(5):
        baseline, task = _config_pair(fold)
        assert _normalized_for_parity(task) == _normalized_for_parity(baseline)
        assert task["model"]["name"] == "carewsm_gated_task_attention"
        assert task["experiment_info"]["params"] == {
            "experiment_name": EXPERIMENT,
            "run_name": f"F_task_attention_fold{fold}",
            "include_time": False,
        }
        assert all(
            callback["params"].get("log_path", LOG_PATH) == LOG_PATH
            for callback in task["callbacks"]
        )
        assert task["logging"][0]["params"]["log_path"] == LOG_PATH
        assert task["logging"][1]["params"]["tracking_uri"] == TRACKING_URI
        assert "weighted_f" not in yaml.safe_dump(task)


def test_task_attention_configs_resolve_registered_model() -> None:
    chimera_plugin.register()
    for fold in range(5):
        _, config = _config_pair(fold)
        model = MODELS.create(config["model"]["name"], **config["model"]["params"])
        assert model.__class__.__name__ == "CAREWSMGatedTaskAttentionModel"


class SyntheticBatch:
    def __init__(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        self.inputs = {"native": values}
        self._mask = mask

    def get_masks(self, name: str) -> torch.Tensor | None:
        return self._mask if name == "sequence_mask" else None


def test_synthetic_forward_backward_matches_weighted_loss_interface() -> None:
    torch.manual_seed(85)
    model = MODELS.create(
        "carewsm_gated_task_attention",
        input_dim=2048,
        family_dims=[1024, 1024],
        hidden_dim=8,
        dropout=0.0,
    )
    values = torch.randn(2, 3, 2048, requires_grad=True)
    sequence_mask = torch.tensor([[True, True, False], [True, True, True]])
    batch = Batch(
        inputs={"native": values},
        targets=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        masks={
            "sequence_mask": sequence_mask,
            "observed_targets": torch.tensor([[True, False], [False, True]]),
        },
    )

    output = model(batch)
    loss = CAREWSMWeightedParkinsonBCELoss(2.358878504672897)(output, batch)
    loss.backward()

    assert tuple(output.preds.shape) == (2, 2)
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_launcher_declares_safe_sequential_preflight() -> None:
    source = LAUNCHER.read_text()
    assert "set -Eeuo pipefail" in source
    assert source.index("F_task_attention_fold0.yaml") < source.index(
        "F_task_attention_fold4.yaml"
    )
    assert source.count('\"$CONFIG_ROOT/F_task_attention_fold') == 5
    assert 'assert_no_collision \"$config\"' in source
    assert 'if [[ \"$mode\" == \"--check\" ]]' in source
    assert "Preflight passed; no training was launched." in source
    assert 'CUDA_VISIBLE_DEVICES=0 \"$CHIMERA\" train' in source
    assert "AMBIGUOUS_COLLISION" not in source
    assert "COLLISION" in source
    assert "audio_representation_pilot_deep008_task_attention" in source
