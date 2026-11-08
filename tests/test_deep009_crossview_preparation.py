from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

import chimera_plugin
from audio.loss.carewsm_masked_bce_loss import CAREWSMMaskedBCELoss
from audio.models.carewsm_representation_pilot_models import (
    CAREWSMCrossViewModel,
    CAREWSMGatedFusionModel,
)
from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS


BASELINE_ROOT = Path("configs/audio_representation_pilot/deep008/weighted_f")
CROSSVIEW_ROOT = Path("configs/audio_representation_pilot/deep009/crossview")
LAUNCHER = Path("scripts/run_audio_deep009_crossview.sh")
LOG_PATH = "logs/audio_representation_pilot/deep009_crossview"
EXPERIMENT = "audio_representation_pilot_deep009_crossview"
TRACKING_URI = "sqlite:///logs/audio_representation_pilot/deep009_crossview/mlflow.db"


def _config_pair(fold: int) -> tuple[dict, dict]:
    baseline = yaml.safe_load(
        (BASELINE_ROOT / f"F_weighted_parkinson_fold{fold}.yaml").read_text()
    )
    crossview = yaml.safe_load(
        (CROSSVIEW_ROOT / f"F_crossview_fold{fold}.yaml").read_text()
    )
    return baseline, crossview


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


def test_five_configs_preserve_weighted_f_semantics() -> None:
    paths = sorted(CROSSVIEW_ROOT.glob("F_crossview_fold[0-4].yaml"))
    assert [path.name for path in paths] == [
        f"F_crossview_fold{fold}.yaml" for fold in range(5)
    ]
    for fold in range(5):
        baseline, crossview = _config_pair(fold)
        assert _normalized_for_parity(crossview) == _normalized_for_parity(baseline)
        assert crossview["model"]["name"] == "carewsm_crossview"
        assert crossview["experiment_info"]["params"] == {
            "experiment_name": EXPERIMENT,
            "run_name": f"F_crossview_fold{fold}",
            "include_time": False,
        }
        assert all(
            callback["params"].get("log_path", LOG_PATH) == LOG_PATH
            for callback in crossview["callbacks"]
        )
        assert crossview["logging"][0]["params"]["log_path"] == LOG_PATH
        assert crossview["logging"][1]["params"]["tracking_uri"] == TRACKING_URI


def test_crossview_resolves_from_chimera_registry() -> None:
    chimera_plugin.register()
    model = MODELS.create(
        "carewsm_crossview",
        input_dim=2048,
        family_dims=[1024, 1024],
        hidden_dim=64,
        dropout=0.0,
    )
    assert isinstance(model, CAREWSMCrossViewModel)


class SyntheticBatch:
    def __init__(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        targets: torch.Tensor | None = None,
        observed: torch.Tensor | None = None,
    ) -> None:
        self.inputs = {"native": values}
        self.targets = targets
        self._masks = {"sequence_mask": mask}
        if observed is not None:
            self._masks["observed_targets"] = observed

    def get_masks(self, name: str) -> torch.Tensor | None:
        return self._masks.get(name)


def _model() -> CAREWSMCrossViewModel:
    model = CAREWSMCrossViewModel(
        input_dim=2048,
        family_dims=[1024, 1024],
        hidden_dim=64,
        dropout=0.0,
    )
    model.eval()
    return model


def test_crossview_has_exact_parameter_count_and_t1_output() -> None:
    model = _model()
    assert sum(parameter.numel() for parameter in model.parameters()) == 278275
    assert sum(parameter.numel() for parameter in model.emotion_interaction.parameters()) == 1024
    assert sum(parameter.numel() for parameter in model.trillsson_interaction.parameters()) == 1024
    assert sum(parameter.numel() for parameter in model.interaction_output.parameters()) == 1024
    assert model.alpha.numel() == 1

    batch = SyntheticBatch(torch.randn(3, 1, 2048), torch.ones(3, 1, dtype=torch.bool))
    output = model(batch)
    assert output.preds.shape == (3, 2)
    assert torch.isfinite(output.preds).all()


def test_crossview_padding_is_invariant_and_all_invalid_is_rejected() -> None:
    torch.manual_seed(101)
    model = _model()
    values = torch.randn(2, 4, 2048)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    output = model(SyntheticBatch(values, mask)).preds

    changed = values.clone()
    changed[0, 2:] = 1000.0
    changed[1, 1:] = -1000.0
    changed_output = model(SyntheticBatch(changed, mask)).preds
    assert torch.allclose(output, changed_output)

    invalid = torch.zeros(1, 3, 2048)
    with pytest.raises(ValueError, match="at least one valid turn"):
        model(SyntheticBatch(invalid, torch.zeros(1, 3, dtype=torch.bool)))


def test_crossview_gradients_flow_through_interaction_and_masked_loss() -> None:
    torch.manual_seed(102)
    model = CAREWSMCrossViewModel(
        input_dim=2048,
        family_dims=[1024, 1024],
        hidden_dim=64,
        dropout=0.0,
    )
    values = torch.randn(3, 3, 2048, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True], [True, False, False]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    observed = torch.tensor([[True, False], [True, True], [False, True]])
    batch = Batch(
        inputs={"native": values},
        targets=targets,
        masks={"sequence_mask": mask, "observed_targets": observed},
    )
    output = model(batch)
    loss = CAREWSMMaskedBCELoss()(output, batch)
    loss.backward()

    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_crossview_missing_labels_are_ignored() -> None:
    model = _model()
    values = torch.randn(2, 2, 2048)
    mask = torch.ones(2, 2, dtype=torch.bool)
    observed = torch.tensor([[True, False], [False, True]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    changed_targets = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    output = model(SyntheticBatch(values, mask))
    first = Batch(
        inputs={"native": values}, targets=targets, masks={"observed_targets": observed}
    )
    second = Batch(
        inputs={"native": values},
        targets=changed_targets,
        masks={"observed_targets": observed},
    )
    loss_first = CAREWSMMaskedBCELoss()(output, first)
    loss_second = CAREWSMMaskedBCELoss()(output, second)
    assert torch.allclose(loss_first, loss_second)


def test_crossview_cpu_bfloat16_autocast_backward() -> None:
    torch.manual_seed(103)
    model = CAREWSMCrossViewModel(
        input_dim=2048,
        family_dims=[1024, 1024],
        hidden_dim=64,
        dropout=0.0,
    )
    values = torch.randn(2, 3, 2048, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(SyntheticBatch(values, mask))
        loss = output.preds.square().mean()
    loss.backward()
    assert torch.isfinite(output.preds).all()
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_historical_gated_fusion_forward_contract_is_unchanged() -> None:
    torch.manual_seed(104)
    values = torch.randn(2, 3, 2048)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    first = CAREWSMGatedFusionModel(2048, [1024, 1024], hidden_dim=64, dropout=0.0)
    second = CAREWSMGatedFusionModel(2048, [1024, 1024], hidden_dim=64, dropout=0.0)
    second.load_state_dict(first.state_dict())
    first.eval()
    second.eval()
    first_output = first(SyntheticBatch(values, mask)).preds
    second_output = second(SyntheticBatch(values, mask)).preds
    assert first_output.shape == (2, 2)
    assert torch.allclose(first_output, second_output)


def test_launcher_is_cpu_check_only_before_training_mode() -> None:
    source = LAUNCHER.read_text()
    assert "set -Eeuo pipefail" in source
    assert source.count('"$CONFIG_ROOT/F_crossview_fold') == 5
    assert source.index("F_crossview_fold0.yaml") < source.index("F_crossview_fold4.yaml")
    assert 'if [[ "$mode" == "--check" ]]' in source
    assert "Preflight passed; no training was launched." in source
    assert "audio_representation_pilot_deep009_crossview" in source
    assert "deep009/crossview" in source
