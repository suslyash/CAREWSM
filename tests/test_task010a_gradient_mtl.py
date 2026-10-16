from __future__ import annotations

import numpy as np
import torch
import pytest
from scipy.optimize import minimize

from chimera_ml.core.batch import Batch
from fusion.loss.gradient_mtl_loss import (
    EPS,
    WSMGradientMTLLoss,
    _ParameterOwnership,
    _inject_parameter_gradients,
    cagrad_two,
    dbmtl_update,
    pcgrad_two,
)
from fusion.loss.r4_ramps_balance_loss import WSMR4RampsBalanceLoss
from fusion.models.av_r3_disease_query import WSMAVR3DiseaseQueryModel
from common.callbacks.wsm_gradient_mtl_callback import WSMGradientMTLCallback


def _cagrad_reference(gd: torch.Tensor, gp: torch.Tensor) -> torch.Tensor:
    grads = torch.stack([gd, gp])
    gram = (grads @ grads.T).double().numpy()
    g0_norm = float(np.sqrt(gram.mean() + EPS))
    x0 = np.array([0.5, 0.5])
    c = 0.5 * g0_norm + EPS

    def objective(x: np.ndarray) -> float:
        linear = (x[None, :] @ gram @ x0[:, None]).item()
        quadratic = (x[None, :] @ gram @ x[:, None]).item()
        return float(linear + c * np.sqrt(quadratic + EPS))

    result = minimize(
        objective, x0, method="SLSQP", bounds=((0, 1), (0, 1)),
        constraints=({"type": "eq", "fun": lambda x: 1.0 - x.sum()},),
        options={"ftol": 1e-12, "maxiter": 100},
    )
    assert result.success
    weights = torch.tensor(result.x, dtype=grads.dtype)
    gw = (grads * weights[:, None]).sum(0)
    return (grads.mean(0) + c / (gw.norm() + EPS) * gw) / 1.25


def _batch() -> Batch:
    torch.manual_seed(77)
    size = 5
    return Batch(
        inputs={
            "audio_cls": torch.randn(size, 768),
            "video": torch.randn(size, 3, 512),
            "pseudo_targets": torch.full((size, 2), float("nan")),
            "pseudo_reliability": torch.zeros(size, 2),
        },
        targets=torch.tensor([[0., 1.], [1., 0.], [0., 1.], [1., 0.], [0., 1.]]),
        masks={
            "video_mask": torch.ones(size, 3, dtype=torch.bool),
            "modality_available": torch.ones(size, 2, dtype=torch.bool),
            "observed_mask": torch.ones(size, 2, dtype=torch.bool),
            "pseudo_accept_mask": torch.zeros(size, 2, dtype=torch.bool),
        },
    )


def test_equal_component_path_matches_frozen_loss_and_gradients() -> None:
    torch.manual_seed(12)
    model = WSMAVR3DiseaseQueryModel()
    model.eval()
    batch = _batch()
    frozen = WSMR4RampsBalanceLoss(mode="equal", pseudo_scale=0.0)
    extension = WSMGradientMTLLoss(method="pcgrad", pseudo_scale=0.0)

    out = model(batch)
    frozen_loss = frozen(out, batch)
    objectives, active = extension._task_objectives(out, batch)
    assert active == [True, True]
    extracted = 0.5 * (objectives[0] + objectives[1])
    assert torch.allclose(frozen_loss, extracted, rtol=0.0, atol=1e-7)
    frozen_loss.backward()
    frozen_grads = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    out = model(batch)
    objectives, _ = extension._task_objectives(out, batch)
    (0.5 * (objectives[0] + objectives[1])).backward()
    for expected, parameter in zip(frozen_grads, model.parameters(), strict=True):
        assert expected is not None and parameter.grad is not None
        assert torch.allclose(expected, parameter.grad, rtol=1e-6, atol=1e-7)


def test_pcgrad_hand_cases_and_task_specific_gradient() -> None:
    aligned_d = torch.tensor([1.0, 2.0])
    aligned_p = torch.tensor([2.0, 4.0])
    got_d, got_p = pcgrad_two(aligned_d, aligned_p)
    assert torch.equal(got_d, aligned_d) and torch.equal(got_p, aligned_p)
    conflict_d = torch.tensor([1.0, 0.0])
    conflict_p = torch.tensor([-1.0, 1.0])
    got_d, got_p = pcgrad_two(conflict_d, conflict_p)
    dot = torch.dot(conflict_d, conflict_p)
    assert torch.allclose(got_d, conflict_d - dot * conflict_p / conflict_p.square().sum())
    assert torch.allclose(got_p, conflict_p - dot * conflict_d / conflict_d.square().sum())
    shared = torch.nn.Parameter(torch.tensor(1.0))
    d_only = torch.nn.Parameter(torch.tensor(2.0))
    p_only = torch.nn.Parameter(torch.tensor(3.0))
    value = _inject_parameter_gradients(
        shared * 0.0, [shared, d_only, p_only],
        [torch.tensor(4.0), torch.tensor(5.0), torch.tensor(6.0)],
    )
    value.backward()
    assert shared.grad.item() == 4.0 and d_only.grad.item() == 5.0 and p_only.grad.item() == 6.0


def test_cagrad_matches_independent_pinned_formula_cases() -> None:
    for gd, gp in [
        (torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])),
        (torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0])),
        (torch.tensor([3.0, -2.0]), torch.tensor([0.5, 4.0])),
    ]:
        got, weights, solved = cagrad_two(gd, gp)
        assert solved and np.isclose(weights.sum(), 1.0)
        assert torch.allclose(got, _cagrad_reference(gd, gp), rtol=1e-6, atol=1e-6)


def test_gradnorm_weights_follow_frozen_update_contract() -> None:
    loss = WSMGradientMTLLoss(method="gradnorm", pseudo_scale=0.0)
    objective_d = torch.tensor(1.0, requires_grad=True)
    objective_p = torch.tensor(1.0, requires_grad=True)
    first = loss._gradnorm_combine([torch.tensor([1.0])], [torch.tensor([1.0])], [0], [objective_d, objective_p])
    assert torch.allclose(loss.gradnorm_weights, torch.ones(2))
    assert torch.allclose(first[0], torch.tensor([2.0]))
    slower_d = torch.tensor(2.0, requires_grad=True)
    steady_p = torch.tensor(1.0, requires_grad=True)
    loss._gradnorm_combine([torch.tensor([1.0])], [torch.tensor([1.0])], [0], [slower_d, steady_p])
    weights = loss.gradnorm_weights
    assert bool((weights > 0).all()) and torch.allclose(weights.sum(), torch.tensor(2.0), atol=1e-6)
    assert weights[0] > weights[1]
    assert loss.gradnorm_alpha == 1.5 and loss.gradnorm_weight_lr == 0.025


def test_dbmtl_algorithm_one_ema_scaling_and_task_locality() -> None:
    d = torch.tensor([3.0, 0.0])
    p = torch.tensor([0.0, 1.0])
    combined, ema = dbmtl_update(d, p, None, beta=0.9, beta_sigma=0.0, step=1)
    assert torch.allclose(ema, torch.stack([d, p]) * 0.1)
    norms = ema.norm(dim=1)
    scaled_norms = norms.max() / (norms + EPS) * norms
    assert torch.allclose(scaled_norms[0], scaled_norms[1], atol=1e-6)
    assert torch.allclose(combined, torch.tensor([0.3, 0.3]), atol=1e-6)
    _, ema2 = dbmtl_update(d, p, ema, beta=0.9, beta_sigma=0.0, step=2)
    assert torch.allclose(ema2, torch.stack([d, p]) * 0.19)
    near_zero, _ = dbmtl_update(torch.zeros(2), torch.zeros(2), None)
    assert torch.isfinite(near_zero).all()


def test_pcgrad_injects_real_r4_parameter_gradients() -> None:
    torch.manual_seed(19)
    model = WSMAVR3DiseaseQueryModel()
    model.train()
    batch = _batch()
    loss = WSMGradientMTLLoss(method="pcgrad", pseudo_scale=0.0)
    loss.bind_model(model)
    scalar = loss(model(batch), batch)
    scalar.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    diagnostics = loss.epoch_diagnostics()
    assert diagnostics["mtl/parameter_tensors_shared"] == 14


@pytest.mark.parametrize("method", ["cagrad", "gradnorm", "dbmtl"])
def test_other_methods_inject_real_r4_parameter_gradients(method: str) -> None:
    torch.manual_seed(23)
    model = WSMAVR3DiseaseQueryModel()
    model.train()
    batch = _batch()
    loss = WSMGradientMTLLoss(method=method, pseudo_scale=0.0)
    loss.bind_model(model)
    scalar = loss(model(batch), batch)
    scalar.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters() if parameter.requires_grad
    )


def test_r4_named_ownership_map_has_exact_structural_counts() -> None:
    model = WSMAVR3DiseaseQueryModel()
    loss = WSMGradientMTLLoss(method="pcgrad", pseudo_scale=0.0)
    loss.bind_model(model)
    assert loss.ownership_counts == {"shared": 14, "depression": 18, "parkinson": 18, "row": 1}
    assert len({entry.name for entry in loss._ownership or []}) == 51


def test_task_specific_gradient_locality_and_shared_finiteness() -> None:
    torch.manual_seed(91)
    model = WSMAVR3DiseaseQueryModel()
    loss = WSMGradientMTLLoss(method="pcgrad", pseudo_scale=0.0)
    loss.bind_model(model)
    batch = _batch()
    output = model(batch)
    objectives, active = loss._task_objectives(output, batch)
    assert active == [True, True]
    parameters = [entry.parameter for entry in loss._ownership or []]
    grad_d = loss._task_gradients(objectives[0], 0, parameters)
    grad_p = loss._task_gradients(objectives[1], 1, parameters)
    for entry, d_grad, p_grad in zip(loss._ownership or [], grad_d, grad_p, strict=True):
        if entry.kind == "depression":
            assert torch.equal(p_grad, torch.zeros_like(p_grad))
        elif entry.kind == "parkinson":
            assert torch.equal(d_grad, torch.zeros_like(d_grad))
        elif entry.kind == "row":
            assert torch.equal(d_grad[1], torch.zeros_like(d_grad[1]))
            assert torch.equal(p_grad[0], torch.zeros_like(p_grad[0]))
        else:
            assert torch.isfinite(d_grad).all() and torch.isfinite(p_grad).all()


@pytest.mark.parametrize("method", ["pcgrad", "cagrad"])
def test_task_specific_zero_gradient_is_not_misclassified_as_shared(method: str) -> None:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = torch.nn.Parameter(torch.tensor([1.0]))
            self.d_only = torch.nn.Parameter(torch.tensor([2.0]))
            self.p_only = torch.nn.Parameter(torch.tensor([3.0]))

    tiny = Tiny()
    loss = WSMGradientMTLLoss(method=method, pseudo_scale=0.0)
    loss._ownership = [
        _ParameterOwnership("shared", tiny.shared, "shared"),
        _ParameterOwnership("d_only", tiny.d_only, "depression"),
        _ParameterOwnership("p_only", tiny.p_only, "parkinson"),
    ]
    loss._shared_indices = [0]
    grad_d = [torch.tensor([2.0]), torch.tensor([3.0]), torch.zeros(1)]
    grad_p = [torch.tensor([-1.0]), torch.zeros(1), torch.tensor([4.0])]
    combined = loss._combine_pair(grad_d, grad_p, [0], [torch.tensor(1.0), torch.tensor(1.0)])
    assert torch.allclose(combined[1], grad_d[1])
    assert torch.allclose(combined[2], grad_p[2])
    assert not torch.allclose(combined[0], 0.5 * (grad_d[0] + grad_p[0]))

def test_shared_named_ownership_map_has_exact_structural_counts() -> None:
    model = WSMAVR3DiseaseQueryModel(task_aware_fusion=False)
    loss = WSMGradientMTLLoss(method="pcgrad", pseudo_scale=0.0)
    loss.bind_model(model)
    assert loss.ownership_counts == {"shared": 23, "depression": 14, "parkinson": 14, "row": 0}
    names = {entry.name for entry in loss._ownership or []}
    assert "task_queries" in names
    assert len(names) == 51


def test_shared_ownership_reaches_both_task_objectives() -> None:
    torch.manual_seed(121)
    model = WSMAVR3DiseaseQueryModel(task_aware_fusion=False)
    loss = WSMGradientMTLLoss(method="pcgrad", pseudo_scale=0.0)
    loss.bind_model(model)
    output = model(_batch())
    objectives, active = loss._task_objectives(output, _batch())
    assert active == [True, True]
    parameters = [entry.parameter for entry in loss._ownership or []]
    raw_d = torch.autograd.grad(objectives[0], parameters, retain_graph=True, allow_unused=True)
    raw_p = torch.autograd.grad(objectives[1], parameters, retain_graph=True, allow_unused=True)
    for entry, d_grad, p_grad in zip(loss._ownership or [], raw_d, raw_p, strict=True):
        if entry.kind == "shared":
            assert d_grad is not None and p_grad is not None
        elif entry.kind == "depression":
            assert d_grad is not None
            assert p_grad is None or torch.equal(p_grad, torch.zeros_like(p_grad))
        elif entry.kind == "parkinson":
            assert p_grad is not None
            assert d_grad is None or torch.equal(d_grad, torch.zeros_like(d_grad))
        else:
            raise AssertionError("Shared fusion must not classify parameters as row-partitioned")


@pytest.mark.parametrize(
    ("task_aware_fusion", "expected"),
    [
        (True, {"shared": 14, "depression": 18, "parkinson": 18, "row": 1}),
        (False, {"shared": 23, "depression": 14, "parkinson": 14, "row": 0}),
    ],
)
def test_gradient_mtl_callback_binds_both_model_modes(task_aware_fusion: bool, expected: dict[str, int]) -> None:
    model = WSMAVR3DiseaseQueryModel(task_aware_fusion=task_aware_fusion)
    loss = WSMGradientMTLLoss(method="pcgrad", pseudo_scale=0.0)
    trainer = type("Trainer", (), {"model": model, "loss_fn": loss})()
    WSMGradientMTLCallback().on_fit_start(trainer)
    assert loss.ownership_counts == expected


class _FakeCallbackLoss:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts
        self.method = "pcgrad"

    @property
    def ownership_counts(self) -> dict[str, int]:
        return self._counts

    def bind_model(self, model: object) -> None:
        del model

    def epoch_diagnostics(self) -> dict[str, float]:
        return {}

    def reset_epoch_diagnostics(self) -> None:
        return None


def test_gradient_mtl_callback_rejects_mismatched_ownership() -> None:
    model = WSMAVR3DiseaseQueryModel(task_aware_fusion=False)
    loss = _FakeCallbackLoss({"shared": 14, "depression": 18, "parkinson": 18, "row": 1})
    trainer = type("Trainer", (), {"model": model, "loss_fn": loss})()
    with pytest.raises(ValueError, match="task_aware_fusion=False"):
        WSMGradientMTLCallback().on_fit_start(trainer)


@pytest.mark.parametrize("value", [None, "false", 1])
def test_gradient_mtl_callback_rejects_missing_or_nonboolean_mode(value: object) -> None:
    model = type("Model", (), {})()
    if value is not None:
        model.task_aware_fusion = value
    loss = _FakeCallbackLoss({"shared": 23, "depression": 14, "parkinson": 14, "row": 0})
    trainer = type("Trainer", (), {"model": model, "loss_fn": loss})()
    with pytest.raises(ValueError, match="boolean task_aware_fusion"):
        WSMGradientMTLCallback().on_fit_start(trainer)


@pytest.mark.parametrize("method", ["pcgrad", "cagrad", "gradnorm", "dbmtl"])
def test_shared_gradient_mtl_methods_inject_finite_gradients(method: str) -> None:
    torch.manual_seed(211)
    model = WSMAVR3DiseaseQueryModel(task_aware_fusion=False)
    model.train()
    loss = WSMGradientMTLLoss(method=method, pseudo_scale=0.0, pcgrad_seed=211)
    loss.bind_model(model)
    scalar = loss(model(_batch()), _batch())
    scalar.backward()
    assert torch.isfinite(scalar)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters() if parameter.requires_grad
    )
    assert loss.ownership_counts == {"shared": 23, "depression": 14, "parkinson": 14, "row": 0}
