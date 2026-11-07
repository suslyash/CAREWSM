"""WSM-inspired multitask balancing losses for the CAREWSM gated fusion model."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import torch
from scipy.optimize import minimize
from torch.nn import functional as F

from chimera_ml.core.registry import LOSSES
from chimera_ml.losses.base import BaseLoss

EPS = 1.0e-8
METHODS = {
    "equal",
    "stch",
    "progress",
    "ra_stch",
    "pcgrad",
    "cagrad",
    "gradnorm",
    "dbmtl",
}


def _flat(values: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.detach().reshape(-1) for value in values])


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    norm = left.norm() * right.norm()
    if float(norm) <= EPS:
        return None
    value = torch.dot(left, right) / norm
    return float(torch.clamp(value, -1.0, 1.0).item())


def pcgrad_two(
    depression: torch.Tensor, parkinson: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project conflicting task gradients before equal-scale aggregation."""
    result_d = depression.clone()
    result_p = parkinson.clone()
    dot = torch.dot(depression, parkinson)
    if float(dot) < 0.0:
        result_d = result_d - dot * parkinson / (parkinson.square().sum() + EPS)
        result_p = result_p - dot * depression / (depression.square().sum() + EPS)
    return result_d, result_p


def cagrad_two(
    depression: torch.Tensor,
    parkinson: torch.Tensor,
    calpha: float = 0.5,
) -> tuple[torch.Tensor, np.ndarray]:
    """Solve the pinned two-task CAGrad simplex problem."""
    gradients = torch.stack((depression, parkinson))
    gram = (gradients @ gradients.T).detach().double().cpu().numpy()
    start = np.full(2, 0.5, dtype=np.float64)
    constant = calpha * float(np.sqrt(gram.mean() + EPS)) + EPS

    def objective(weights: np.ndarray) -> float:
        linear = weights.reshape(1, -1) @ gram @ start.reshape(-1, 1)
        quadratic = weights.reshape(1, -1) @ gram @ weights.reshape(-1, 1)
        return float(linear.item() + constant * np.sqrt(quadratic.item() + EPS))

    result = minimize(
        objective,
        start,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        constraints={"type": "eq", "fun": lambda weights: 1.0 - weights.sum()},
        method="SLSQP",
        options={"ftol": 1.0e-12, "maxiter": 100},
    )
    if not result.success:
        raise RuntimeError(f"CAGrad simplex solve failed: {result.message}")
    weights = torch.as_tensor(result.x, device=gradients.device, dtype=gradients.dtype)
    combined = (gradients * weights[:, None]).sum(dim=0)
    lam = constant / (combined.norm().item() + EPS)
    combined = (gradients.mean(dim=0) + lam * combined) / (1.0 + calpha**2)
    return combined, result.x.copy()


def dbmtl_update(
    depression: torch.Tensor,
    parkinson: torch.Tensor,
    previous: torch.Tensor | None,
    step: int,
    beta: float = 0.9,
    beta_sigma: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the Algorithm-1 DB-MTL EMA and max-norm balancing."""
    current = torch.stack((depression, parkinson))
    previous = torch.zeros_like(current) if previous is None else previous
    decay = beta / (float(step) ** beta_sigma)
    ema = current + decay * (previous - current)
    norms = ema.norm(dim=1)
    scale = norms.max() / (norms + EPS)
    return (scale[:, None] * ema).sum(dim=0), ema


def _observed_targets(batch: Any) -> torch.Tensor:
    observed = batch.get_masks("observed_targets")
    if observed is None:
        observed = batch.inputs.get("observed_targets")
    if observed is None and batch.meta is not None:
        observed = batch.meta.get("observed_targets")
        if observed is None and "sample_meta" in batch.meta:
            observed = torch.stack(
                [item["observed_targets"] for item in batch.meta["sample_meta"]]
            )
    if observed is None:
        raise ValueError("Batch does not contain observed-target masks")
    return observed


def _inject_gradients(
    value: torch.Tensor,
    parameters: list[torch.Tensor],
    gradients: list[torch.Tensor],
) -> torch.Tensor:
    class GradientInjection(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, scalar: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
            del params
            ctx.gradients = gradients
            return scalar.detach()

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[None, ...]:
            return (None, *(grad_output * gradient for gradient in ctx.gradients))

    return GradientInjection.apply(value, *parameters)


class CAREWSMStageGMTLLoss(BaseLoss):
    """Observed-label weighted BCE with WSM-inspired balancing controllers."""

    def __init__(
        self,
        method: str,
        parkinson_pos_weight: float,
        tau: float = 0.1,
        progress_temperature: float = 0.25,
        progress_reference_depression: float = 0.5,
        progress_reference_parkinson: float = 0.5,
        controller_ema: float = 0.8,
        grad_ema: float = 0.9,
        weight_min: float = 0.2,
        weight_max: float = 0.8,
        pcgrad_seed: int = 5,
        calpha: float = 0.5,
        gradnorm_alpha: float = 1.5,
        gradnorm_weight_lr: float = 0.025,
        db_beta: float = 0.9,
        db_beta_sigma: float = 0.0,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)}")
        if parkinson_pos_weight <= 0 or not math.isfinite(parkinson_pos_weight):
            raise ValueError("parkinson_pos_weight must be finite and positive")
        if tau <= 0 or progress_temperature <= 0:
            raise ValueError("tau and progress_temperature must be positive")
        self.method = method
        self.pos_weight = float(parkinson_pos_weight)
        self.tau = float(tau)
        self.progress_temperature = float(progress_temperature)
        self.references = (
            float(progress_reference_depression),
            float(progress_reference_parkinson),
        )
        self.controller_ema = float(controller_ema)
        self.grad_ema = float(grad_ema)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        if not 0.0 <= self.controller_ema < 1.0 or not 0.0 <= self.grad_ema < 1.0:
            raise ValueError("EMA values must be in [0, 1)")
        if not 0.0 <= self.weight_min <= self.weight_max <= 1.0:
            raise ValueError("weight bounds must satisfy 0 <= min <= max <= 1")
        self._rng = random.Random(pcgrad_seed)
        self.calpha = float(calpha)
        self.gradnorm_alpha = float(gradnorm_alpha)
        self.gradnorm_weight_lr = float(gradnorm_weight_lr)
        self.db_beta = float(db_beta)
        self.db_beta_sigma = float(db_beta_sigma)
        self.weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
        self.last_progress = [0.0, 0.0]
        self.last_scores = [float("nan"), float("nan")]
        self.grad_norm_ema: list[float | None] = [None, None]
        self.grad_cos_ema: float | None = None
        self._parameters: list[torch.Tensor] | None = None
        self._shared: list[int] = []
        self._private_d: list[int] = []
        self._private_p: list[int] = []
        self._gradnorm_weights: torch.nn.Parameter | None = None
        self._gradnorm_optimizer: torch.optim.Adam | None = None
        self._initial_losses: torch.Tensor | None = None
        self._db_buffer: torch.Tensor | None = None
        self._db_step = 0
        self._epoch_stats: defaultdict[str, list[float]] = defaultdict(list)

    def bind_model(self, model: Any) -> None:
        """Bind actual gated-fusion ownership before the first training batch."""
        parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self._parameters = [parameter for _, parameter in parameters]
        self._shared = []
        self._private_d = []
        self._private_p = []
        for index, (name, parameter) in enumerate(parameters):
            if name.startswith(("projections.", "gates.")):
                self._shared.append(index)
            elif name.startswith("heads.depression."):
                self._private_d.append(index)
            elif name.startswith("heads.parkinson."):
                self._private_p.append(index)
            else:
                raise ValueError(f"unclassified gated-fusion parameter: {name}")
        if len(self._shared) != 12 or len(self._private_d) != 6 or len(self._private_p) != 6:
            raise ValueError(
                "unexpected gated-fusion ownership: "
                f"shared={len(self._shared)}, depression={len(self._private_d)}, "
                f"parkinson={len(self._private_p)}"
            )

    @property
    def ownership_counts(self) -> dict[str, int]:
        return {
            "shared": len(self._shared),
            "depression": len(self._private_d),
            "parkinson": len(self._private_p),
        }

    def reset_epoch_diagnostics(self) -> None:
        self._epoch_stats = defaultdict(list)

    def _record(self, **values: float | None) -> None:
        for key, value in values.items():
            if value is not None and math.isfinite(float(value)):
                self._epoch_stats[key].append(float(value))

    def epoch_diagnostics(self) -> dict[str, float]:
        return {
            f"mtl/{key}": sum(values) / len(values)
            for key, values in self._epoch_stats.items()
            if values
        }

    def _components(self, output: Any, batch: Any) -> tuple[list[torch.Tensor], list[bool]]:
        logits = output.preds
        observed = _observed_targets(batch)
        observed = observed.to(device=logits.device, dtype=torch.bool)
        targets = batch.targets.to(device=logits.device, dtype=logits.dtype)
        objectives: list[torch.Tensor] = []
        active: list[bool] = []
        weights = torch.tensor(
            [1.0, self.pos_weight], device=logits.device, dtype=logits.dtype
        )
        for task in range(2):
            mask = observed[:, task]
            active.append(bool(mask.any()))
            if active[-1]:
                objectives.append(
                    F.binary_cross_entropy_with_logits(
                        logits[mask, task],
                        targets[mask, task],
                        pos_weight=weights[task],
                    )
                )
            else:
                objectives.append(logits[:, task].sum() * 0.0)
        return objectives, active

    def _task_gradients(
        self, objective: torch.Tensor, task: int
    ) -> list[torch.Tensor]:
        if self._parameters is None:
            raise RuntimeError("loss must be bound to a model before training")
        raw = torch.autograd.grad(
            objective, self._parameters, retain_graph=True, allow_unused=True
        )
        result = []
        for index, gradient in enumerate(raw):
            value = torch.zeros_like(self._parameters[index]) if gradient is None else gradient
            if index in self._private_d and task == 1:
                value = torch.zeros_like(value)
            if index in self._private_p and task == 0:
                value = torch.zeros_like(value)
            result.append(value)
        return result

    def _combine_gradients(
        self,
        grad_d: list[torch.Tensor],
        grad_p: list[torch.Tensor],
        objectives: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        assert self._parameters is not None
        if self.method == "gradnorm":
            shared_d = _flat(grad_d[index] for index in self._shared)
            shared_p = _flat(grad_p[index] for index in self._shared)
            weights = self._compute_gradnorm_weights(shared_d, shared_p, objectives)
            return [
                weights[0] * depression + weights[1] * parkinson
                for depression, parkinson in zip(grad_d, grad_p, strict=True)
            ]

        shared_d = _flat(grad_d[index] for index in self._shared)
        shared_p = _flat(grad_p[index] for index in self._shared)
        pre_cosine = _cosine(shared_d, shared_p)
        self._record(
            pre_cosine=pre_cosine,
            conflict=float(pre_cosine is not None and pre_cosine < 0.0),
        )
        if self.method == "pcgrad":
            order = [0, 1]
            self._rng.shuffle(order)
            transformed_d, transformed_p = pcgrad_two(shared_d, shared_p)
            shared = 0.5 * (transformed_d + transformed_p)
            self._record(post_cosine=_cosine(transformed_d, transformed_p))
        elif self.method == "cagrad":
            shared, weights = cagrad_two(shared_d, shared_p, self.calpha)
            self._record(
                cagrad_weight_depression=float(weights[0]),
                cagrad_weight_parkinson=float(weights[1]),
                cagrad_adjustment=float(
                    (shared - 0.5 * (shared_d + shared_p)).norm().item()
                ),
            )
        else:
            return [
                0.5 * (left + right)
                for left, right in zip(grad_d, grad_p, strict=True)
            ]

        chunks: dict[int, torch.Tensor] = {}
        offset = 0
        for index in self._shared:
            width = grad_d[index].numel()
            chunks[index] = shared[offset : offset + width].reshape_as(grad_d[index])
            offset += width
        result = []
        for index in range(len(self._parameters)):
            if index in self._shared:
                result.append(chunks[index])
            elif index in self._private_d:
                result.append(grad_d[index])
            else:
                result.append(grad_p[index])
        return result

    def _compute_gradnorm_weights(
        self,
        shared_d: torch.Tensor,
        shared_p: torch.Tensor,
        objectives: list[torch.Tensor],
    ) -> torch.Tensor:
        reference = objectives[0]
        if self._gradnorm_weights is None:
            self._gradnorm_weights = torch.nn.Parameter(
                torch.ones(2, device=reference.device, dtype=reference.dtype)
            )
            self._gradnorm_optimizer = torch.optim.Adam(
                [self._gradnorm_weights], lr=self.gradnorm_weight_lr
            )
        assert self._gradnorm_optimizer is not None and self._gradnorm_weights is not None
        weights_before = self._gradnorm_weights.detach().clone()
        losses = torch.stack([objective.detach() for objective in objectives])
        norms = torch.stack([shared_d.norm(), shared_p.norm()]).detach()
        if self._initial_losses is None:
            self._initial_losses = losses.clone()
        else:
            rates = losses / self._initial_losses.clamp_min(EPS)
            rates = rates / rates.mean().clamp_min(EPS)
            scaled = self._gradnorm_weights * norms
            target = (scaled.mean() * rates.pow(self.gradnorm_alpha)).detach()
            auxiliary = (scaled - target).abs().sum()
            self._gradnorm_optimizer.zero_grad(set_to_none=True)
            auxiliary.backward()
            self._gradnorm_optimizer.step()
            with torch.no_grad():
                self._gradnorm_weights.clamp_(min=EPS)
                self._gradnorm_weights.mul_(2.0 / self._gradnorm_weights.sum().clamp_min(EPS))
        self._record(
            weight_depression=float(weights_before[0].item()),
            weight_parkinson=float(weights_before[1].item()),
            gradient_norm_depression=float(norms[0].item()),
            gradient_norm_parkinson=float(norms[1].item()),
        )
        return weights_before

    def __call__(self, output: Any, batch: Any) -> torch.Tensor:
        objectives, active = self._components(output, batch)
        if active == [True, False]:
            return objectives[0]
        if active == [False, True]:
            return objectives[1]
        if not any(active):
            return output.preds.sum() * 0.0
        self._record(
            task_loss_depression=float(objectives[0].detach().item()),
            task_loss_parkinson=float(objectives[1].detach().item()),
        )
        if self.method == "equal":
            self._record(scalar_weight_depression=0.5, scalar_weight_parkinson=0.5)
            return 0.5 * (objectives[0] + objectives[1])
        if self.method == "stch":
            self._record(scalar_weight_depression=0.5, scalar_weight_parkinson=0.5)
            return self.tau * torch.logsumexp(
                torch.stack(
                    (0.5 * objectives[0] / self.tau, 0.5 * objectives[1] / self.tau)
                ),
                dim=0,
            )
        if not output.preds.requires_grad:
            return 0.5 * (objectives[0] + objectives[1])
        if self.method in {"progress", "ra_stch"}:
            weights = self.weights.to(device=output.preds.device, dtype=output.preds.dtype)
            if self.method == "progress":
                self._record(
                    scalar_weight_depression=float(weights[0].item()),
                    scalar_weight_parkinson=float(weights[1].item()),
                )
                return (weights * torch.stack(objectives)).sum()
            if self._parameters is None:
                raise RuntimeError("loss must be bound to a model before RA-STCH")
            grad_d = self._task_gradients(objectives[0], 0)
            grad_p = self._task_gradients(objectives[1], 1)
            shared_d = _flat(grad_d[index] for index in self._shared)
            shared_p = _flat(grad_p[index] for index in self._shared)
            norm_values = (float(shared_d.norm().item()), float(shared_p.norm().item()))
            cosine = _cosine(shared_d, shared_p)
            for task, value in enumerate(norm_values):
                previous = self.grad_norm_ema[task]
                self.grad_norm_ema[task] = (
                    value
                    if previous is None
                    else self.grad_ema * previous + (1.0 - self.grad_ema) * value
                )
            if cosine is not None:
                self.grad_cos_ema = (
                    cosine
                    if self.grad_cos_ema is None
                    else self.grad_ema * self.grad_cos_ema
                    + (1.0 - self.grad_ema) * cosine
                )
            self._record(
                gradient_norm_depression=norm_values[0],
                gradient_norm_parkinson=norm_values[1],
                gradient_cosine=cosine,
                gradient_norm_ema_depression=self.grad_norm_ema[0],
                gradient_norm_ema_parkinson=self.grad_norm_ema[1],
                gradient_cosine_ema=self.grad_cos_ema,
            )
            self._record(
                scalar_weight_depression=float(weights[0].item()),
                scalar_weight_parkinson=float(weights[1].item()),
            )
            return self.tau * torch.logsumexp(
                weights * torch.stack(objectives) / self.tau, dim=0
            )
        if self._parameters is None:
            raise RuntimeError("loss must be bound to a model before gradient MTL")
        grad_d = self._task_gradients(objectives[0], 0)
        grad_p = self._task_gradients(objectives[1], 1)
        if self.method == "dbmtl":
            self._db_step += 1
            log_d = self._task_gradients(torch.log(objectives[0] + EPS), 0)
            log_p = self._task_gradients(torch.log(objectives[1] + EPS), 1)
            shared_d = _flat(log_d[index] for index in self._shared)
            shared_p = _flat(log_p[index] for index in self._shared)
            shared, self._db_buffer = dbmtl_update(
                shared_d, shared_p, self._db_buffer, self._db_step, self.db_beta, self.db_beta_sigma
            )
            combined = self._combine_from_shared(log_d, log_p, shared)
            self._record(
                dbmtl_norm_depression=float(shared_d.norm().item()),
                dbmtl_norm_parkinson=float(shared_p.norm().item()),
                dbmtl_cosine=_cosine(shared_d, shared_p),
            )
        else:
            combined = self._combine_gradients(grad_d, grad_p, objectives)
        self._record(
            parameter_tensors_shared=float(len(self._shared)),
            parameter_tensors_depression_only=float(len(self._private_d)),
            parameter_tensors_parkinson_only=float(len(self._private_p)),
        )
        return _inject_gradients(0.5 * (objectives[0] + objectives[1]), self._parameters, combined)

    def _combine_from_shared(
        self,
        grad_d: list[torch.Tensor],
        grad_p: list[torch.Tensor],
        shared: torch.Tensor,
    ) -> list[torch.Tensor]:
        assert self._parameters is not None
        result = []
        offset = 0
        for index in range(len(self._parameters)):
            if index in self._shared:
                width = grad_d[index].numel()
                result.append(shared[offset : offset + width].reshape_as(grad_d[index]))
                offset += width
            elif index in self._private_d:
                result.append(grad_d[index])
            else:
                result.append(grad_p[index])
        return result

    def update_controller(self, depression_score: float, parkinson_score: float) -> None:
        scores = (float(depression_score), float(parkinson_score))
        self.last_scores = list(scores)
        progress = torch.tensor(
            [
                max(-1.0, min(1.0, (score - reference) / (1.0 - reference + EPS)))
                for score, reference in zip(scores, self.references, strict=True)
            ],
            dtype=torch.float64,
        )
        self.last_progress = [float(value) for value in progress]
        if self.method not in {"progress", "ra_stch"}:
            return

        raw = torch.exp(-progress / self.progress_temperature)
        if self.method == "ra_stch":
            norms = torch.tensor(
                [value if value is not None else 1.0 for value in self.grad_norm_ema],
                dtype=torch.float64,
            )
            mean_norm = norms.mean()
            cosine = self.grad_cos_ema if self.grad_cos_ema is not None else 0.0
            exponent = 1.0 + max(0.0, -cosine)
            raw = raw * (mean_norm / (norms + EPS)).clamp(0.5, 2.0).pow(exponent)
            # CAREWSM has no pseudo-label reliability signal; the declared
            # neutral reliability factor is therefore one for both tasks.

        bounded = raw / raw.sum().clamp_min(EPS)
        bounded = bounded.clamp(self.weight_min, self.weight_max)
        bounded = bounded / bounded.sum().clamp_min(EPS)
        if self.method == "progress":
            self.weights = bounded
        else:
            self.weights = (
                self.controller_ema * self.weights
                + (1.0 - self.controller_ema) * bounded
            )
            self.weights = self.weights.clamp(self.weight_min, self.weight_max)
            self.weights = self.weights / self.weights.sum().clamp_min(EPS)
        self._record(
            controller_weight_depression=float(self.weights[0]),
            controller_weight_parkinson=float(self.weights[1]),
        )


@LOSSES.register("carewsm_stageg_mtl")
def carewsm_stageg_mtl(
    context: Any | None = None, **params: Any
) -> CAREWSMStageGMTLLoss:
    del context
    return CAREWSMStageGMTLLoss(**params)
