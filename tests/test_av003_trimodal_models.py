"""No-training AV-003 fusion model checks."""

import pytest
import torch

import chimera_plugin
from audio.loss.carewsm_masked_bce_loss import CAREWSMMaskedBCELoss
from audio.metrics.carewsm_audio_metrics import CAREWSMAudioClassificationMetric
from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput

MODELS_UNDER_TEST = (
    "carewsm_text_guided_dual_gated",
    "carewsm_trimodal_cross_attention",
    "carewsm_shared_private_trimodal",
)


def make_batch(lengths=(1, 1, 1), padded_length=None):
    torch.manual_seed(11)
    dimensions = {"audio": 4, "video": 5, "text": 3}
    max_length = padded_length or max(lengths)
    inputs = {
        modality: torch.randn(2, max_length, dimension)
        for modality, dimension in dimensions.items()
    }
    masks = {}
    for modality, length in zip(("audio", "video", "text"), lengths):
        values = torch.arange(max_length).unsqueeze(0) < torch.tensor([[length], [length]])
        masks[f"{modality}_lengths"] = torch.tensor([length, length])
        if modality == "video":
            masks["video_valid_mask"] = values
    masks["observed_targets"] = torch.ones(2, 2, dtype=torch.bool)
    return Batch(inputs=inputs, targets=torch.zeros(2, 2), masks=masks)


@pytest.mark.parametrize("name", MODELS_UNDER_TEST)
def test_registered_models_forward_backward_and_two_heads(name):
    chimera_plugin.register()
    model = MODELS.create(
        name, audio_dim=4, video_dim=5, text_dim=3, hidden_dim=12, dropout=0.0
    )
    batch = make_batch()
    output = model(batch)
    assert isinstance(output, ModelOutput)
    assert output.preds.shape == (2, 2)
    assert torch.isfinite(output.preds).all()
    output.preds.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert not torch.equal(output.preds[:, 0], output.preds[:, 1])


@pytest.mark.parametrize("name", MODELS_UNDER_TEST)
def test_each_modality_affects_predictions(name):
    torch.manual_seed(7)
    model = MODELS.create(name, audio_dim=4, video_dim=5, text_dim=3, hidden_dim=12, dropout=0.0)
    model.eval()
    batch = make_batch()
    original = model(batch).preds
    for modality in ("audio", "video", "text"):
        changed = make_batch()
        perturbation = 0.37 * torch.arange(
            changed.inputs[modality].numel(), dtype=torch.float32
        ).reshape_as(changed.inputs[modality])
        changed.inputs[modality] = changed.inputs[modality] + perturbation
        assert not torch.allclose(original, model(changed).preds)


@pytest.mark.parametrize("name", MODELS_UNDER_TEST)
def test_padding_is_ignored(name):
    torch.manual_seed(19)
    model = MODELS.create(name, audio_dim=4, video_dim=5, text_dim=3, hidden_dim=12, dropout=0.0)
    model.eval()
    short = make_batch(lengths=(2, 2, 2), padded_length=2)
    padded = make_batch(lengths=(2, 2, 2), padded_length=4)
    for modality in ("audio", "video", "text"):
        padded.inputs[modality][:, :2] = short.inputs[modality]
        padded.inputs[modality][:, 2:] = 1000.0
    assert torch.allclose(model(short).preds, model(padded).preds, atol=1e-5)


def test_cross_attention_has_static_and_sequence_paths():
    model = MODELS.create(
        "carewsm_trimodal_cross_attention",
        audio_dim=4, video_dim=5, text_dim=3, hidden_dim=12, dropout=0.0,
    )
    static_output = model(make_batch())
    sequence_output = model(make_batch(lengths=(3, 2, 4), padded_length=4))
    assert static_output.aux["fusion_path"] == "static_fallback"
    assert sequence_output.aux["fusion_path"] == "cross_attention"


def test_existing_loss_and_metrics_accept_model_output():
    batch = make_batch()
    output = ModelOutput(preds=torch.zeros(2, 2), aux={})
    loss = CAREWSMMaskedBCELoss()(output, batch)
    assert torch.isfinite(loss)
    metric = CAREWSMAudioClassificationMetric()
    metric.update(output, batch)
    assert "depression/macro_f1" in metric.compute()


@pytest.mark.parametrize("name", MODELS_UNDER_TEST)
def test_invalid_dimensions_and_missing_modalities_fail(name):
    model = MODELS.create(name, audio_dim=4, video_dim=5, text_dim=3, hidden_dim=12)
    batch = make_batch()
    batch.inputs["audio"] = torch.zeros(2, 1, 99)
    with pytest.raises(ValueError, match="audio"):
        model(batch)
    batch = make_batch()
    del batch.inputs["text"]
    with pytest.raises(ValueError, match="text"):
        model(batch)
