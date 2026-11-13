from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from audio.callbacks.carewsm_stageg_mtl_callback import CAREWSMStageGMTLCallback
from audio.loss.carewsm_masked_bce_loss import CAREWSMMaskedBCELoss
from audio.loss.carewsm_stageg_mtl_loss import (
    CAREWSMStageGMTLLoss,
    cagrad_two,
    dbmtl_update,
    pcgrad_two,
)
from audio.loss.carewsm_weighted_bce_loss import CAREWSMWeightedParkinsonBCELoss
from audio.models.carewsm_representation_pilot_models import CAREWSMGatedFusionModel


def _batch(observed: torch.Tensor | None = None) -> Batch:
    if observed is None:
        observed = torch.ones(4, 2, dtype=torch.bool)
    return Batch(
        inputs={"native": torch.randn(4, 1, 2048)},
        targets=torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        masks={"sequence_mask": torch.ones(4, 1, dtype=torch.bool), "observed_targets": observed},
    )


def _output(logits: torch.Tensor) -> ModelOutput:
    return ModelOutput(preds=logits, aux={})


def test_equal_matches_weighted_f_forward_and_parameter_gradients() -> None:
    torch.manual_seed(12)
    model = CAREWSMGatedFusionModel(2048, [1024, 1024], 64, 0.1)
    model.eval()
    batch = _batch()
    output = model(batch)
    equal = CAREWSMStageGMTLLoss("equal", 2.35)
    weighted = CAREWSMWeightedParkinsonBCELoss(2.35)
    equal_value = equal(output, batch)
    weighted_value = weighted(output, batch)
    assert torch.allclose(equal_value, weighted_value, rtol=0.0, atol=1.0e-7)
    equal_value.backward()
    equal_grads = [parameter.grad.detach().clone() for parameter in model.parameters()]
    model.zero_grad(set_to_none=True)
    weighted(model(batch), batch).backward()
    for expected, parameter in zip(equal_grads, model.parameters(), strict=True):
        assert parameter.grad is not None
        assert torch.allclose(expected, parameter.grad, rtol=1.0e-6, atol=1.0e-7)


def test_missing_label_masks_and_inactive_head_isolation() -> None:
    for observed, active_task in (
        (torch.ones(4, 2, dtype=torch.bool), None),
        (torch.tensor([[True, False]] * 4), 0),
        (torch.tensor([[False, True]] * 4), 1),
        (torch.zeros(4, 2, dtype=torch.bool), None),
    ):
        torch.manual_seed(21)
        model = CAREWSMGatedFusionModel(2048, [1024, 1024], 64, 0.1)
        loss = CAREWSMStageGMTLLoss("equal", 2.35)
        batch = _batch(observed)
        value = loss(model(batch), batch)
        assert torch.isfinite(value)
        value.backward()
        depression = [p for n, p in model.named_parameters() if n.startswith("heads.depression.")]
        parkinson = [p for n, p in model.named_parameters() if n.startswith("heads.parkinson.")]
        if active_task == 0:
            assert all(
                parameter.grad is not None and torch.all(parameter.grad == 0)
                for parameter in parkinson
            )
        elif active_task == 1:
            assert all(
                parameter.grad is not None and torch.all(parameter.grad == 0)
                for parameter in depression
            )
        elif not observed.any():
            assert all(
                parameter.grad is not None and torch.all(parameter.grad == 0)
                for parameter in model.parameters()
            )


def test_parameter_ownership_and_gradnorm_weights_apply_to_private_heads() -> None:
    model = CAREWSMGatedFusionModel(2048, [1024, 1024], 64, 0.1)
    loss = CAREWSMStageGMTLLoss("gradnorm", 2.35)
    loss.bind_model(model)
    assert loss.ownership_counts == {"shared": 12, "depression": 6, "parkinson": 6}
    loss._gradnorm_weights = torch.nn.Parameter(torch.tensor([2.0, 0.5]))
    loss._gradnorm_optimizer = torch.optim.Adam([loss._gradnorm_weights], lr=0.025)
    with torch.no_grad():
        loss._initial_losses = torch.ones(2)
    gradients_d = [torch.ones_like(parameter) for parameter in loss._parameters or []]
    gradients_p = [torch.ones_like(parameter) * 3.0 for parameter in loss._parameters or []]
    for index in loss._private_p:
        gradients_d[index].zero_()
    for index in loss._private_d:
        gradients_p[index].zero_()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loss, "_gradnorm_weights", torch.nn.Parameter(torch.tensor([2.0, 0.5])))
        patch.setattr(
            loss,
            "_gradnorm_optimizer",
            torch.optim.Adam([loss._gradnorm_weights], lr=0.025),
        )
        patch.setattr(loss, "_compute_gradnorm_weights", lambda *args: torch.tensor([2.0, 0.5]))
        combined = loss._combine_gradients(
            gradients_d,
            gradients_p,
            [torch.tensor(1.0), torch.tensor(1.0)],
        )
    for index in loss._shared:
        assert torch.allclose(combined[index], torch.full_like(combined[index], 3.5))
    for index in loss._private_d:
        assert torch.allclose(combined[index], torch.full_like(combined[index], 2.0))
    for index in loss._private_p:
        assert torch.allclose(combined[index], torch.full_like(combined[index], 1.5))


def test_stch_scalarization_and_gradients() -> None:
    logits = torch.tensor([[0.2, -0.4]], requires_grad=True)
    batch = _batch(torch.ones(4, 2, dtype=torch.bool))
    batch = Batch(
        inputs=batch.inputs,
        targets=batch.targets[:1],
        masks={"observed_targets": torch.ones(1, 2, dtype=torch.bool)},
    )
    value = CAREWSMStageGMTLLoss("stch", 2.35, tau=0.1)(_output(logits), batch)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(logits.grad).all()


def test_pcgrad_cagrad_dbmtl_invariants() -> None:
    aligned_d, aligned_p = pcgrad_two(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0]))
    assert torch.equal(aligned_d, torch.tensor([1.0, 2.0]))
    assert torch.equal(aligned_p, torch.tensor([2.0, 4.0]))
    conflict_d, conflict_p = pcgrad_two(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0]))
    assert torch.dot(conflict_d, conflict_p) >= -1.0e-7
    combined, weights = cagrad_two(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    assert torch.isfinite(combined).all() and abs(float(weights.sum()) - 1.0) < 1.0e-8
    combined, ema = dbmtl_update(torch.tensor([3.0, 0.0]), torch.tensor([0.0, 1.0]), None, 1)
    assert torch.isfinite(combined).all() and torch.isfinite(ema).all()


def test_progress_and_ra_controller_calculations() -> None:
    progress = CAREWSMStageGMTLLoss("progress", 2.35)
    progress.update_controller(0.9, 0.5)
    direct_target = progress.weights.clone()
    assert not torch.allclose(direct_target, torch.tensor([0.5, 0.5], dtype=torch.float64))
    assert torch.allclose(progress.weights, direct_target)

    ra = CAREWSMStageGMTLLoss("ra_stch", 2.35)
    ra.grad_norm_ema = [1.0, 1.0]
    ra.grad_cos_ema = 0.0
    ra.update_controller(0.7, 0.7)
    neutral = ra.weights.clone()
    ra.weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    ra.grad_norm_ema = [4.0, 1.0]
    ra.grad_cos_ema = -1.0
    ra.update_controller(0.7, 0.7)
    assert not torch.allclose(neutral, ra.weights)


def test_callback_requires_both_validation_corpora_and_ignores_test_feedback() -> None:
    callback = CAREWSMStageGMTLCallback()
    loss = CAREWSMStageGMTLLoss("progress", 2.35)
    loss._record(example=1.0)
    calls: list[tuple[float, float]] = []
    loss.update_controller = lambda depression, parkinson: calls.append((depression, parkinson))
    logger = SimpleNamespace(log_metrics=lambda metrics, step: setattr(logger, "metrics", metrics))
    trainer = SimpleNamespace(loss_fn=loss, mlflow_logger=logger)
    logs = {
        "val_care/depression/macro_recall": 0.6,
        "val_wsm/depression/macro_recall": 0.8,
        "val_care/parkinson/macro_recall": 0.4,
        "val_wsm/parkinson/macro_recall": 0.6,
        "test_care/depression/macro_recall": 0.0,
        "test_wsm/depression/macro_recall": 0.0,
    }
    callback.on_epoch_end(trainer, 1, logs)
    assert calls == [(0.7, 0.5)]
    assert "mtl/example" in logger.metrics
    with pytest.raises(KeyError):
        callback.on_epoch_end(trainer, 2, {"val_care/depression/macro_recall": 0.5})


@pytest.mark.parametrize("method", ("equal", "pcgrad", "dbmtl"))
def test_static_gated_mtl_shared_and_private_gradient_ownership(method: str) -> None:
    """Static fused model must retain both task-specific heads under gradient MTL."""
    from audio.models.carewsm_static_models import CAREWSMStaticGatedFusionModel

    torch.manual_seed(41)
    model = CAREWSMStaticGatedFusionModel(
        input_dim=2048, family_dims=[1024, 1024], hidden_dim=32, dropout=0.0
    )
    loss = CAREWSMStageGMTLLoss(method, 2.35)
    loss.bind_model(model)
    assert loss.ownership_counts == {"shared": 12, "depression": 6, "parkinson": 6}
    batch = _batch()
    value = loss(model(batch), batch)
    assert torch.isfinite(value)
    value.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

