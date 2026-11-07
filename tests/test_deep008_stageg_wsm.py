from __future__ import annotations

import torch

from chimera_ml.core.batch import Batch
from audio.loss.carewsm_stageg_mtl_loss import (
    CAREWSMStageGMTLLoss,
    cagrad_two,
    dbmtl_update,
    pcgrad_two,
)
from audio.models.carewsm_representation_pilot_models import CAREWSMGatedFusionModel


def _batch(observed: torch.Tensor | None = None) -> Batch:
    if observed is None:
        observed = torch.ones(4, 2, dtype=torch.bool)
    return Batch(
        inputs={"native": torch.randn(4, 1, 2048)},
        targets=torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        masks={"sequence_mask": torch.ones(4, 1, dtype=torch.bool), "observed_targets": observed},
    )


def test_gradient_invariants() -> None:
    aligned_d, aligned_p = pcgrad_two(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0]))
    assert torch.equal(aligned_d, torch.tensor([1.0, 2.0]))
    assert torch.equal(aligned_p, torch.tensor([2.0, 4.0]))
    conflict_d, conflict_p = pcgrad_two(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0]))
    assert torch.dot(conflict_d, conflict_p) >= -1.0e-7
    combined, weights = cagrad_two(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    assert torch.isfinite(combined).all()
    assert abs(float(weights.sum()) - 1.0) < 1.0e-8
    combined, ema = dbmtl_update(torch.tensor([3.0, 0.0]), torch.tensor([0.0, 1.0]), None, 1)
    assert torch.isfinite(combined).all() and torch.isfinite(ema).all()


def test_all_methods_bind_and_backward() -> None:
    for method in ("equal", "stch", "progress", "ra_stch", "pcgrad", "cagrad", "gradnorm", "dbmtl"):
        torch.manual_seed(21)
        model = CAREWSMGatedFusionModel(2048, [1024, 1024], 64, 0.1)
        loss = CAREWSMStageGMTLLoss(method, 2.35)
        loss.bind_model(model)
        value = loss(model(_batch()), _batch())
        value.backward()
        assert torch.isfinite(value)
        assert loss.ownership_counts == {"shared": 12, "depression": 6, "parkinson": 6}
        assert all(parameter.grad is not None for parameter in model.parameters())


def test_missing_label_and_single_active_task() -> None:
    observed = torch.tensor([[True, False], [True, False], [True, False], [True, False]])
    model = CAREWSMGatedFusionModel(2048, [1024, 1024], 64, 0.1)
    loss = CAREWSMStageGMTLLoss("pcgrad", 2.35)
    loss.bind_model(model)
    value = loss(model(_batch(observed)), _batch(observed))
    value.backward()
    assert torch.isfinite(value)
    assert all(parameter.grad is not None for parameter in model.parameters())
