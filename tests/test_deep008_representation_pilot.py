from pathlib import Path
import pytest
import torch
import yaml

import chimera_plugin
from audio.data.carewsm_audio_dataset import CAREWSMAudioDataset
from audio.models.carewsm_representation_pilot_models import (
    CAREWSMAttentiveMLPModel,
    CAREWSMGatedFusionModel,
    CAREWSMGatedTaskAttentionModel,
)
from chimera_ml.core.registry import MODELS


CONFIG_ROOT = Path("configs/audio_representation_pilot/deep008")


class Batch:
    def __init__(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        self.inputs = {"native": values}
        self._mask = mask

    def get_masks(self, name: str) -> torch.Tensor | None:
        return self._mask if name == "sequence_mask" else None


def test_original_model_module_is_baseline_only() -> None:
    import audio.models.carewsm_audio_models as original

    assert not hasattr(original, "MaskedAttentionPool")
    assert not hasattr(original, "CAREWSMAttentiveMLPModel")
    assert not hasattr(original, "CAREWSMGatedFusionModel")
    assert hasattr(original, "carewsm_pooled_mlp")


def test_pilot_registrations_are_idempotent() -> None:
    chimera_plugin.register()
    chimera_plugin.register()
    assert MODELS.create("carewsm_attentive_mlp", input_dim=4, hidden_dim=4)
    assert MODELS.create(
        "carewsm_gated_fusion", input_dim=2048, family_dims=[1024, 1024], hidden_dim=4
    )
    assert MODELS.create(
        "carewsm_gated_task_attention",
        input_dim=2048,
        family_dims=[1024, 1024],
        hidden_dim=4,
    )


def test_pilot_configs_are_isolated() -> None:
    paths = sorted(CONFIG_ROOT.glob("*.yaml"))
    assert len(paths) == 6
    assert not Path("configs/audio_deep_features/deep008_pilot").exists()
    for path in paths:
        config = yaml.safe_load(path.read_text())
        assert config["experiment_info"]["params"]["experiment_name"] == (
            "audio_representation_pilot_deep008"
        )
        assert config["logging"][1]["params"]["tracking_uri"] == (
            "sqlite:///logs/audio_representation_pilot/mlflow.db"
        )
        assert all(
            callback["params"].get("log_path") != "logs/audio_deep_features"
            for callback in config["callbacks"]
        )
        assert config["logging"][0]["params"]["log_path"] == (
            "logs/audio_representation_pilot"
        )


def test_pooling_supports_single_turn_and_variable_lengths(tmp_path: Path) -> None:
    path = tmp_path / "feature.npz"
    np = pytest.importorskip("numpy")
    np.savez(path, embed=np.arange(12, dtype=np.float32).reshape(3, 4), turn_ids=[1, 2, 3])
    record = {
        "corpus": "CARE",
        "sample_id": "sample",
        "profile_id": "profile",
        "emotionmsp_embeds_status": "VALID",
        "feature_paths": {"emotionmsp_embeds": str(path)},
        "depression_observed": True,
        "parkinson_observed": True,
        "depression": 0.0,
        "parkinson": 1.0,
    }
    spec = {"emotionmsp_embeds": {"format": "npz", "embedding_keys": ["embed"]}}
    mean = CAREWSMAudioDataset([record], "emotionmsp_embeds", spec, pooling="mean")[0]
    mean_std = CAREWSMAudioDataset(
        [record], "emotionmsp_embeds", spec, pooling="mean_std"
    )[0]
    assert tuple(mean["inputs"]["native"].shape) == (1, 4)
    assert tuple(mean_std["inputs"]["native"].shape) == (1, 8)


def test_pilot_models_mask_padding_and_backpropagate() -> None:
    values = torch.randn(2, 4, 1024, requires_grad=True)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    attentive = CAREWSMAttentiveMLPModel(1024, 64)
    output = attentive(Batch(values, mask))
    loss = output.preds.sum()
    loss.backward()
    assert tuple(output.preds.shape) == (2, 2)

    gated_values = torch.randn(2, 4, 2048, requires_grad=True)
    gated = CAREWSMGatedFusionModel(2048, [1024, 1024], 64)
    gated_output = gated(Batch(gated_values, mask))
    gated_output.preds.sum().backward()
    assert tuple(gated_output.preds.shape) == (2, 2)


def test_gated_fusion_rejects_wrong_family_contract() -> None:
    with pytest.raises(ValueError):
        CAREWSMGatedFusionModel(3075, [1024, 3, 1024, 1024])


def _task_attention_model(hidden_dim: int = 8) -> CAREWSMGatedTaskAttentionModel:
    model = CAREWSMGatedTaskAttentionModel(
        2048, [1024, 1024], hidden_dim=hidden_dim, dropout=0.0
    )
    model.eval()
    return model


def test_task_attention_matches_gated_fusion_for_one_turn() -> None:
    torch.manual_seed(30)
    gated = CAREWSMGatedFusionModel(2048, [1024, 1024], hidden_dim=8, dropout=0.0)
    attentive = _task_attention_model()
    attentive.load_state_dict(gated.state_dict(), strict=False)
    values = torch.randn(2, 1, 2048)
    batch = Batch(values, torch.ones(2, 1, dtype=torch.bool))

    with torch.no_grad():
        gated_logits = gated(batch).preds
        attentive_output = attentive(batch)

    assert torch.allclose(attentive_output.preds, gated_logits)
    assert torch.all(attentive_output.aux["attention_weights"] == 1.0)


def test_task_attention_is_task_specific_and_mask_safe() -> None:
    torch.manual_seed(31)
    model = _task_attention_model()
    with torch.no_grad():
        model.depression_attention.weight.zero_()
        model.depression_attention.weight[0, 0] = 1.0
        model.parkinson_attention.weight.zero_()
        model.parkinson_attention.weight[0, 0] = -1.0
    values = torch.randn(2, 4, 2048)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    output = model(Batch(values, mask))
    attention = output.aux["attention_weights"]

    assert tuple(output.preds.shape) == (2, 2)
    assert not torch.allclose(attention[:, 0], attention[:, 1])
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2, 2))
    assert torch.all(attention[:, :, 3] == 0.0)
    assert torch.all(attention[1, :, 2:] == 0.0)

    changed = values.clone()
    changed[0, 3:] = 1000.0
    changed[1, 2:] = 1000.0
    changed_output = model(Batch(changed, mask))
    assert torch.allclose(output.preds, changed_output.preds)


def test_task_attention_rejects_all_invalid_sequences() -> None:
    model = _task_attention_model()
    values = torch.randn(2, 3, 2048)
    mask = torch.tensor([[False, False, False], [True, False, False]])

    with pytest.raises(ValueError, match="at least one valid turn"):
        model(Batch(values, mask))


def test_task_attention_gradients_reach_scorers_and_heads() -> None:
    torch.manual_seed(32)
    model = CAREWSMGatedTaskAttentionModel(
        2048, [1024, 1024], hidden_dim=8, dropout=0.0
    )
    values = torch.randn(2, 3, 2048, requires_grad=True)
    mask = torch.tensor([[True, True, True], [True, True, False]])

    model(Batch(values, mask)).preds.sum().backward()

    for parameter in (
        model.depression_attention.weight,
        model.parkinson_attention.weight,
        *model.heads.depression.parameters(),
        *model.heads.parkinson.parameters(),
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
