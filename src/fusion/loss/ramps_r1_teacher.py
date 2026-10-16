"""Deterministic RAMPS R1 binary teacher calibration utilities."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _validate_binary_inputs(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.as_tensor(logits, dtype=torch.float64).reshape(-1)
    targets = torch.as_tensor(targets, dtype=torch.float64).reshape(-1)
    if logits.numel() == 0 or logits.shape != targets.shape:
        raise ValueError("binary calibration inputs must be non-empty and shape-matched")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("binary calibration inputs must be finite")
    if not bool(torch.logical_or(targets == 0, targets == 1).all()):
        raise ValueError("binary calibration targets must be 0/1")
    return logits, targets


def binary_nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    logits, targets = _validate_binary_inputs(logits, targets)
    return float(F.binary_cross_entropy_with_logits(logits, targets).item())


def binary_brier(logits: torch.Tensor, targets: torch.Tensor) -> float:
    logits, targets = _validate_binary_inputs(logits, targets)
    probabilities = torch.sigmoid(logits)
    return float(torch.mean((probabilities - targets) ** 2).item())


def binary_ece(logits: torch.Tensor, targets: torch.Tensor, bins: int = 15) -> float:
    if bins <= 0:
        raise ValueError("bins must be positive")
    logits, targets = _validate_binary_inputs(logits, targets)
    probabilities = torch.sigmoid(logits)
    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=probabilities.dtype)
    total = torch.zeros((), dtype=probabilities.dtype)
    count = probabilities.numel()
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        support = int(selected.sum().item())
        if support:
            total = total + (support / count) * (
                probabilities[selected].mean() - targets[selected].mean()
            ).abs()
    return float(total.item())


def fit_binary_temperature(
    raw_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    minimum: float = 0.05,
    maximum: float = 20.0,
    max_iter: int = 100,
) -> dict[str, float]:
    """Fit log-temperature by deterministic full-batch BCE minimization."""
    logits, labels = _validate_binary_inputs(raw_logits, targets)
    if not (0.0 < minimum <= maximum):
        raise ValueError("temperature bounds are invalid")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.5, max_iter=max_iter,
        tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        objective = F.binary_cross_entropy_with_logits(logits / temperature, labels)
        objective.backward()
        return objective

    optimizer.step(closure)
    unconstrained = float(torch.exp(log_temperature.detach()).item())
    deployed = min(max(unconstrained, minimum), maximum)
    return {
        "unconstrained_temperature": unconstrained,
        "temperature": deployed,
        "clamped": bool(deployed != unconstrained),
    }


def select_class_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive: bool,
    precision_target: float = 0.90,
    min_support: int = 10,
    threshold_step: float = 0.01,
) -> dict[str, Any]:
    """Select the fixed-support precision threshold for one class side."""
    probabilities = torch.as_tensor(probabilities, dtype=torch.float64).reshape(-1)
    labels = torch.as_tensor(labels, dtype=torch.bool).reshape(-1)
    if probabilities.shape != labels.shape or not bool(torch.isfinite(probabilities).all()):
        raise ValueError("threshold inputs must be finite and shape-matched")
    if not (0.0 < precision_target <= 1.0) or min_support <= 0 or threshold_step <= 0:
        raise ValueError("invalid threshold policy")
    start, stop = (0.50, 0.99) if positive else (0.01, 0.50)
    candidates = [
        round(start + index * threshold_step, 10)
        for index in range(int(round((stop - start) / threshold_step)) + 1)
    ]
    audits: list[dict[str, Any]] = []
    for threshold in candidates:
        selected = probabilities >= threshold if positive else probabilities <= threshold
        support = int(selected.sum().item())
        true_class = labels if positive else ~labels
        correct = int((selected & true_class).sum().item())
        precision = float(correct / support) if support else 0.0
        class_support = int(true_class.sum().item())
        coverage = float(correct / class_support) if class_support else 0.0
        audits.append({
            "threshold": threshold, "support": support, "correct": correct,
            "precision": precision, "coverage": coverage,
        })
    eligible = [
        audit for audit in audits
        if audit["support"] >= min_support and audit["precision"] >= precision_target
    ]
    chosen = (min(eligible, key=lambda x: x["threshold"]) if positive
              else max(eligible, key=lambda x: x["threshold"])) if eligible else None
    return {
        "enabled": chosen is not None,
        "threshold": None if chosen is None else chosen["threshold"],
        "support": 0 if chosen is None else chosen["support"],
        "precision": None if chosen is None else chosen["precision"],
        "coverage": 0.0 if chosen is None else chosen["coverage"],
        "precision_target": float(precision_target),
        "min_support": int(min_support),
        "threshold_step": float(threshold_step),
        "candidate_audit": audits,
    }
