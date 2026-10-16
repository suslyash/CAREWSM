"""Frozen R4 task-balancing loss for the RAMPS interaction bundle."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from chimera_ml.core.registry import LOSSES
from chimera_ml.losses.base import BaseLoss


def _value(batch: Any, name: str) -> torch.Tensor:
    if isinstance(batch, Mapping):
        value = batch.get(name)
    elif name == "targets":
        value = getattr(batch, "targets", None)
    else:
        value = None
        getter = getattr(batch, "get_masks", None)
        if callable(getter):
            value = getter(name)
        if value is None and isinstance(getattr(batch, "inputs", None), Mapping):
            value = batch.inputs.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"batch must provide tensor {name!r}")
    return value


class WSMR4RampsBalanceLoss(BaseLoss):
    """Observed/pseudo R3 objective with frozen equal, STCH, progress, and RA-STCH modes."""

    MODES = {"equal", "stch", "progress", "ra_stch"}

    def __init__(self, mode: str, pseudo_scale: float = 0.0, aux_weight: float = 0.25,
                 agreement_weight: float = 0.10, tau: float = 0.1,
                 progress_temperature: float = 0.25,
                 progress_reference_depression: float = 0.697035,
                 progress_reference_parkinson: float = 0.852104,
                 controller_ema: float = 0.8, grad_ema: float = 0.9,
                 reliability_ema: float = 0.9, weight_min: float = 0.2,
                 weight_max: float = 0.8, eps: float = 1e-8,
                 training_task: str = "both") -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        if training_task not in {"both", "depression", "parkinson"}:
            raise ValueError("training_task must be one of ['both', 'depression', 'parkinson']")
        for name, value in (("pseudo_scale", pseudo_scale), ("aux_weight", aux_weight),
                            ("agreement_weight", agreement_weight), ("tau", tau),
                            ("progress_temperature", progress_temperature),
                            ("progress_reference_depression", progress_reference_depression),
                            ("progress_reference_parkinson", progress_reference_parkinson),
                            ("controller_ema", controller_ema), ("grad_ema", grad_ema),
                            ("reliability_ema", reliability_ema), ("weight_min", weight_min),
                            ("weight_max", weight_max), ("eps", eps)):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= float(pseudo_scale) <= 1.0 or float(aux_weight) < 0 or float(agreement_weight) < 0:
            raise ValueError("loss scales must be non-negative and pseudo_scale must be in [0, 1]")
        if float(tau) <= 0 or float(progress_temperature) <= 0 or float(eps) < 0:
            raise ValueError("tau, progress_temperature, and eps must be positive/non-negative")
        if not 0.0 <= float(controller_ema) < 1.0 or not 0.0 <= float(grad_ema) < 1.0 or not 0.0 <= float(reliability_ema) < 1.0:
            raise ValueError("EMA decays must be in [0, 1)")
        if not 0.0 <= float(weight_min) <= float(weight_max) <= 1.0:
            raise ValueError("weight bounds must satisfy 0 <= min <= max <= 1")
        self.mode = mode
        self.training_task = training_task
        self.pseudo_scale = float(pseudo_scale)
        self.aux_weight = float(aux_weight)
        self.agreement_weight = float(agreement_weight)
        self.tau = float(tau)
        self.progress_temperature = float(progress_temperature)
        self.references = (float(progress_reference_depression), float(progress_reference_parkinson))
        self.controller_ema = float(controller_ema)
        self.grad_ema = float(grad_ema)
        self.reliability_ema_decay = float(reliability_ema)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self.eps = float(eps)
        self.weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
        self.grad_norm_ema: list[float | None] = [None, None]
        self.grad_cos_ema: float | None = None
        self.reliability_ema: list[float | None] = [None, None]
        self.last_progress = [0.0, 0.0]
        self.last_batch_grad_norms: list[float | None] = [None, None]
        self.last_batch_grad_cosine: float | None = None

    @staticmethod
    def _bounded_normalize(values: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
        values = values.detach().to(dtype=torch.float64)
        values = values / values.sum().clamp_min(torch.finfo(values.dtype).eps)
        values = values.clamp(minimum, maximum)
        return (values / values.sum().clamp_min(torch.finfo(values.dtype).eps)).detach()

    def _components(self, task_logits: torch.Tensor, task: int, targets: torch.Tensor,
                    observed: torch.Tensor, accept: torch.Tensor, pseudo: torch.Tensor,
                    reliability: torch.Tensor, audio_aux: torch.Tensor, video_aux: torch.Tensor,
                    audio_valid: torch.Tensor, video_valid: torch.Tensor) -> tuple[torch.Tensor, bool]:
        obs = observed[:, task]
        pseudo_mask = accept[:, task] & ~obs
        if bool(obs.any()):
            y = targets[:, task]
            if not bool(torch.isfinite(y[obs]).all()) or not bool(((y[obs] == 0) | (y[obs] == 1)).all()):
                raise ValueError("observed targets must be finite binary 0/1")
            observed_loss = F.binary_cross_entropy_with_logits(task_logits[obs, task], y[obs], reduction="mean")
        else:
            observed_loss = task_logits[:, task].sum() * 0.0
        if bool(pseudo_mask.any()):
            p = pseudo[:, task][pseudo_mask]
            r = reliability[:, task][pseudo_mask]
            if not bool(torch.isfinite(p).all()) or not bool(((p >= 0) & (p <= 1)).all()):
                raise ValueError("accepted pseudo targets must be finite in [0, 1]")
            if not bool(torch.isfinite(r).all()) or not bool(((r >= 0) & (r <= 1)).all()):
                raise ValueError("accepted pseudo reliability must be finite in [0, 1]")
            pseudo_loss = (r * F.binary_cross_entropy_with_logits(task_logits[pseudo_mask, task], p, reduction="none")).sum() / (r.sum() + self.eps)
        else:
            pseudo_loss = task_logits[:, task].sum() * 0.0
        eligible_audio = obs & audio_valid[:, task]
        eligible_video = obs & video_valid[:, task]
        count = int(eligible_audio.sum()) + int(eligible_video.sum())
        if count:
            y = targets[:, task]
            aux_sum = F.binary_cross_entropy_with_logits(audio_aux[:, task][eligible_audio], y[eligible_audio], reduction="sum")
            aux_sum = aux_sum + F.binary_cross_entropy_with_logits(video_aux[:, task][eligible_video], y[eligible_video], reduction="sum")
            aux_loss = aux_sum / (count + self.eps)
        else:
            aux_loss = task_logits[:, task].sum() * 0.0
        agree = obs & audio_valid[:, task] & video_valid[:, task]
        agreement = ((torch.sigmoid(audio_aux[:, task][agree]) - torch.sigmoid(video_aux[:, task][agree])) ** 2).mean() if bool(agree.any()) else task_logits[:, task].sum() * 0.0
        active = bool(obs.any() or pseudo_mask.any())
        return observed_loss + self.pseudo_scale * pseudo_loss + self.aux_weight * aux_loss + self.agreement_weight * agreement, active

    def _update_diagnostics(self, objectives: list[torch.Tensor], active: list[bool], features_audio: torch.Tensor,
                            features_video: torch.Tensor, pseudo_mask: torch.Tensor, reliability: torch.Tensor) -> None:
        norms: list[float | None] = [None, None]
        grads: list[torch.Tensor | None] = [None, None]
        task_indices = (0, 1) if self.training_task == "both" else ((0,) if self.training_task == "depression" else (1,))
        both_active = self.training_task == "both" and active == [True, True]
        if self.mode == "ra_stch" and (both_active or self.training_task != "both") and features_audio.requires_grad and features_video.requires_grad:
            for position, task in enumerate(task_indices):
                objective = objectives[position]
                try:
                    ga, gv = torch.autograd.grad(objective, (features_audio, features_video), retain_graph=True, create_graph=False, allow_unused=True)
                    parts = [x.detach().reshape(-1) for x in (ga, gv) if x is not None]
                    if parts:
                        grad = torch.cat(parts)
                        if bool(torch.isfinite(grad).all()):
                            grads[task] = grad
                            norms[task] = float(torch.linalg.vector_norm(grad).item())
                except RuntimeError:
                    pass
        self.last_batch_grad_norms = norms
        if self.mode == "ra_stch" and both_active and grads[0] is not None and grads[1] is not None and grads[0].numel() == grads[1].numel():
            cosine = F.cosine_similarity(grads[0].unsqueeze(0), grads[1].unsqueeze(0)).item()
            if math.isfinite(cosine):
                self.last_batch_grad_cosine = float(max(-1.0, min(1.0, cosine)))
                decay = self.grad_ema
                self.grad_cos_ema = self.last_batch_grad_cosine if self.grad_cos_ema is None else decay * self.grad_cos_ema + (1 - decay) * self.last_batch_grad_cosine
        else:
            self.last_batch_grad_cosine = None
        for task in task_indices:
            if self.mode == "ra_stch" and (both_active or self.training_task != "both") and norms[task] is not None:
                decay = self.grad_ema
                self.grad_norm_ema[task] = norms[task] if self.grad_norm_ema[task] is None else decay * self.grad_norm_ema[task] + (1 - decay) * norms[task]
            accepted = pseudo_mask[:, task]
            if bool(accepted.any()):
                value = float(reliability[:, task][accepted].detach().mean().item())
                decay = self.reliability_ema_decay
                self.reliability_ema[task] = value if self.reliability_ema[task] is None else decay * self.reliability_ema[task] + (1 - decay) * value

    def __call__(self, output: Any, batch: Any) -> torch.Tensor:
        logits = output.preds
        if logits.ndim != 2 or tuple(logits.shape[1:]) != (2,):
            raise ValueError("R4 logits must have shape [B, 2]")
        targets = _value(batch, "targets").to(device=logits.device, dtype=logits.dtype)
        observed = _value(batch, "observed_mask").to(device=logits.device, dtype=torch.bool)
        accept = _value(batch, "pseudo_accept_mask").to(device=logits.device, dtype=torch.bool)
        pseudo = _value(batch, "pseudo_targets").to(device=logits.device, dtype=logits.dtype).detach()
        reliability = _value(batch, "pseudo_reliability").to(device=logits.device, dtype=logits.dtype).detach()
        if bool((accept & observed).any()):
            raise ValueError("pseudo acceptance overlaps observed truth")
        neutral = ~accept
        if bool(torch.isfinite(pseudo[neutral]).any()) or bool((reliability[neutral] != 0).any()):
            raise ValueError("rejected/observed pseudo fields must be NaN/zero")
        audio_aux = output.aux["audio_aux_logits"]
        video_aux = output.aux["video_aux_logits"]
        audio_valid = output.aux["audio_aux_valid"].to(device=logits.device, dtype=torch.bool)
        video_valid = output.aux["video_aux_valid"].to(device=logits.device, dtype=torch.bool)
        if self.training_task == "both":
            objectives, active = [], []
            for task in range(2):
                objective, task_active = self._components(logits, task, targets, observed, accept, pseudo, reliability, audio_aux, video_aux, audio_valid, video_valid)
                objectives.append(objective)
                active.append(task_active)
            if not any(active):
                raise ValueError("R4 batch has neither observed nor accepted-pseudo supervision")
            if active == [True, False]:
                total = objectives[0]
            elif active == [False, True]:
                total = objectives[1]
            else:
                if self.mode == "equal":
                    total = 0.5 * objectives[0] + 0.5 * objectives[1]
                elif self.mode == "stch":
                    total = self.tau * torch.logsumexp(torch.stack([0.5 * objectives[0] / self.tau, 0.5 * objectives[1] / self.tau]), dim=0)
                elif self.mode == "progress":
                    weights = self.weights.to(device=logits.device, dtype=logits.dtype).detach()
                    total = (weights * torch.stack(objectives)).sum()
                else:
                    weights = self.weights.to(device=logits.device, dtype=logits.dtype).detach()
                    total = self.tau * torch.logsumexp(weights * torch.stack(objectives) / self.tau, dim=0)
        else:
            task = 0 if self.training_task == "depression" else 1
            objective, task_active = self._components(logits, task, targets, observed, accept, pseudo, reliability, audio_aux, video_aux, audio_valid, video_valid)
            if not task_active:
                objective = logits[:, task].sum() * 0.0
            objectives, active = [objective], [task_active]
            total = objective
        self._update_diagnostics(objectives, active, output.aux["features_audio"], output.aux["features_video"], accept & ~observed, reliability)
        return total

    def update_controller(self, depression_score: float, parkinson_score: float) -> None:
        scores = (float(depression_score), float(parkinson_score))
        progress = torch.tensor([max(-1.0, min(1.0, (score - ref) / (1.0 - ref + self.eps))) for score, ref in zip(scores, self.references)], dtype=torch.float64)
        self.last_progress = [float(x) for x in progress]
        if self.mode not in {"progress", "ra_stch"}:
            return
        raw = torch.exp(-progress / self.progress_temperature)
        if self.mode == "ra_stch":
            norms = torch.tensor([x if x is not None else 1.0 for x in self.grad_norm_ema], dtype=torch.float64)
            mean_norm = norms.mean()
            exponent = 1.0 + max(0.0, -(self.grad_cos_ema if self.grad_cos_ema is not None else 0.0))
            raw = raw * (mean_norm / (norms + self.eps)).clamp(0.5, 2.0).pow(exponent)
            reliability = torch.tensor([0.5 if x is None else x for x in self.reliability_ema], dtype=torch.float64)
            raw = raw * (0.5 + 0.5 * reliability)
        target = self._bounded_normalize(raw, self.weight_min, self.weight_max)
        if self.mode == "progress":
            self.weights = target
        else:
            self.weights = self._bounded_normalize(self.controller_ema * self.weights + (1.0 - self.controller_ema) * target, self.weight_min, self.weight_max)


@LOSSES.register("wsm_r4_ramps_balance_loss")
def wsm_r4_ramps_balance_loss(context: Any | None = None, **params: Any) -> WSMR4RampsBalanceLoss:
    del context
    return WSMR4RampsBalanceLoss(**params)
