from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from common.callbacks.wsm_segment_callback import compute_sparse_two_task_metrics


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts/common/evaluate_postclosure_shared_progress_test.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("task012a_evaluator", EVALUATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_closed_test_exits_before_all_side_effects(tmp_path, monkeypatch):
    evaluator = load_evaluator()
    calls: list[str] = []

    def sentinel(*args, **kwargs):
        calls.append("side_effect")
        raise AssertionError("closed Test path reached a forbidden side effect")

    monkeypatch.setattr(evaluator, "make_datamodule", sentinel)
    monkeypatch.setattr(evaluator, "model_for", sentinel)
    monkeypatch.setattr(evaluator, "evaluate_one", sentinel)
    args = SimpleNamespace(
        preflight=str(tmp_path / "missing-preflight.json"),
        marker=str(tmp_path / "fresh.marker"),
        output=str(tmp_path / "fresh-output.json"),
        firewall_sha=None,
        pseudo_cache_path=str(tmp_path / "missing-cache.pt"),
    )
    with pytest.raises(RuntimeError, match="TASK-012A is closed"):
        evaluator.test_pass(args)
    assert calls == []
    assert not (tmp_path / "fresh.marker").exists()
    assert not (tmp_path / "fresh-output.json").exists()


def test_closed_interlock_is_first_test_statement():
    tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "test_pass")
    first = function.body[0]
    assert isinstance(first, ast.Raise)
    assert "closed" in ast.unparse(first).lower()


def test_post_evaluation_zero_boundary_and_observed_mask():
    logits = torch.tensor([[0.0, -0.1], [-0.1, 0.0], [0.1, -0.1]])
    targets = torch.tensor([[1.0, float("nan")], [float("nan"), 0.0], [0.0, float("nan")]])
    observed = torch.tensor([[True, False], [False, True], [True, False]])
    metrics = compute_sparse_two_task_metrics(logits, targets, observed, "postcheck")
    assert metrics["postcheck/depression/tp"] == 1.0
    assert metrics["postcheck/depression/fp"] == 1.0
    assert metrics["postcheck/parkinson/tn"] == 0.0
    assert metrics["postcheck/parkinson/fp"] == 1.0


def test_post_evaluation_sample_ci_and_paired_arithmetic():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mean = float(values.mean())
    sample_sd = float(values.std(unbiased=True))
    multiplier = 2.7764451051977987
    half_width = multiplier * sample_sd / (5.0**0.5)
    assert mean == 3.0
    assert abs(sample_sd - (2.5**0.5)) < 1e-7
    assert abs((mean - half_width) - 1.0367568385) < 1e-7
    deltas = torch.tensor([0.2, -0.1, 0.3, 0.0, 0.1])
    assert abs(float(deltas.mean()) - 0.1) < 1e-7
    assert int((deltas > 0).sum()) == 3
