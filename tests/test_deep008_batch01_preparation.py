from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
import yaml


BASELINE_ROOT = Path("configs/audio_representation_pilot/deep008/weighted_f")

import chimera_plugin
from audio.loss.carewsm_batch01_losses import (
    CAREWSMCorpusBalancedWeightedBCELoss,
    CAREWSMWeightedDepressionParkinsonBCELoss,
)
from audio.loss.carewsm_masked_bce_loss import CAREWSMMaskedBCELoss
from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import LOSSES
from chimera_ml.core.types import ModelOutput


CONFIG_ROOT = Path("configs/audio_representation_pilot/deep008/batch01")


def _batch(
    corpora: list[str],
    targets: torch.Tensor,
    observed: torch.Tensor,
) -> Batch:
    return Batch(
        inputs={"native": torch.zeros(len(corpora), 1, 2)},
        targets=targets,
        masks={"observed_targets": observed},
        meta={"sample_meta": [{"corpus": corpus} for corpus in corpora]},
    )


def _output(logits: torch.Tensor) -> ModelOutput:
    return ModelOutput(preds=logits, aux={})


def test_h1_equalizes_available_corpus_losses_within_each_task() -> None:
    logits = torch.tensor([[0.0, 0.0], [2.0, -2.0], [-2.0, 2.0], [0.0, 0.0]])
    targets = torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    observed = torch.ones(4, 2, dtype=torch.bool)
    batch = _batch(["CARE", "CARE", "WSM", "WSM"], targets, observed)
    value = CAREWSMCorpusBalancedWeightedBCELoss(2.0)(_output(logits), batch)

    weights = torch.tensor([1.0, 2.0])
    task_losses = []
    for task in range(2):
        group_losses = []
        for corpus in ("CARE", "WSM"):
            mask = torch.tensor([item == corpus for item in ["CARE", "CARE", "WSM", "WSM"]])
            group_losses.append(
                F.binary_cross_entropy_with_logits(
                    logits[mask, task], targets[mask, task], pos_weight=weights[task]
                )
            )
        task_losses.append(torch.stack(group_losses).mean())
    expected = torch.stack(task_losses).mean()
    assert torch.allclose(value, expected)


def test_h1_omits_missing_corpus_and_label_groups() -> None:
    logits = torch.tensor([[0.3, -0.2], [-0.4, 0.7]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    observed = torch.tensor([[True, False], [False, True]])
    batch = _batch(["CARE", "WSM"], targets, observed)
    value = CAREWSMCorpusBalancedWeightedBCELoss(2.0)(_output(logits), batch)
    expected = torch.stack(
        [
            F.binary_cross_entropy_with_logits(logits[:1, 0], targets[:1, 0]),
            F.binary_cross_entropy_with_logits(
                logits[1:, 1], targets[1:, 1], pos_weight=torch.tensor(2.0)
            ),
        ]
    ).mean()
    assert torch.allclose(value, expected)
    value.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    no_labels = _batch(
        ["CARE", "WSM"],
        targets,
        torch.zeros(2, 2, dtype=torch.bool),
    )
    zero = CAREWSMCorpusBalancedWeightedBCELoss(2.0)(_output(logits.detach()), no_labels)
    assert zero.item() == 0.0


def test_h2_applies_both_positive_weights_and_preserves_parkinson_weight() -> None:
    logits = torch.tensor([[0.2, -0.4], [-0.3, 0.7]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    observed = torch.ones(2, 2, dtype=torch.bool)
    batch = _batch(["CARE", "WSM"], targets, observed)
    value = CAREWSMWeightedDepressionParkinsonBCELoss(1.75, 2.35)(
        _output(logits), batch
    )
    expected = CAREWSMMaskedBCELoss.compute(
        logits,
        targets,
        observed,
        pos_weight=torch.tensor([1.75, 2.35]),
    )
    assert torch.allclose(value, expected)
    value.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_batch01_losses_resolve_and_all_twenty_configs_parse() -> None:
    chimera_plugin.register()
    assert LOSSES.create(
        "carewsm_corpus_balanced_weighted_bce",
        parkinson_pos_weight=2.35,
    ).__class__.__name__ == "CAREWSMCorpusBalancedWeightedBCELoss"
    assert LOSSES.create(
        "carewsm_weighted_depression_parkinson_bce",
        depression_pos_weight=1.75,
        parkinson_pos_weight=2.35,
    ).__class__.__name__ == "CAREWSMWeightedDepressionParkinsonBCELoss"

    paths = sorted(CONFIG_ROOT.glob("H[1-4]_*.yaml"))
    assert len(paths) == 20
    for path in paths:
        config = yaml.safe_load(path.read_text())
        assert config["model"]["name"] == "carewsm_gated_fusion"
        assert config["experiment_info"]["params"]["experiment_name"] == (
            "audio_representation_pilot_deep008_batch01"
        )
        assert config["data"]["params"]["feature_family"] == [
            "emotionmsp_embeds",
            "trillsson4_1",
        ]
        assert config["seed"] == 5
        assert config["train"]["params"]["epochs"] == 50
        assert config["callbacks"][1]["params"]["monitor"] == (
            "val_mean_corpora_tasks/macro_recall"
        )
        assert config["loss"]["name"] in {
            "carewsm_corpus_balanced_weighted_bce",
            "carewsm_weighted_depression_parkinson_bce",
            "carewsm_weighted_parkinson_bce",
        }


def _scientific_parity(config: dict, baseline: dict) -> tuple[dict, dict]:
    def scrub(value: dict) -> dict:
        value = yaml.safe_load(yaml.safe_dump(value, sort_keys=False))
        value["experiment_info"]["params"] = {
            "experiment_name": "EXPERIMENT",
            "run_name": "RUN",
            "include_time": value["experiment_info"]["params"]["include_time"],
        }
        for callback in value["callbacks"]:
            callback["params"].pop("log_path", None)
        for logger in value["logging"]:
            logger["params"].pop("log_path", None)
            logger["params"].pop("tracking_uri", None)
        value["model"]["params"]["dropout"] = 0.1
        value["model"]["params"]["hidden_dim"] = 64
        value["loss"] = {
            "name": "BASELINE_LOSS",
            "params": {
                "parkinson_pos_weight": value["loss"]["params"]["parkinson_pos_weight"],
            },
        }
        return value

    return scrub(config), scrub(baseline)


def test_configs_preserve_weighted_f_semantics_except_approved_changes() -> None:
    expected_depression = [
        1.759119861030689,
        1.747422680412371,
        1.748143917761279,
        1.739749430523918,
        1.740846681922197,
    ]
    for path in sorted(CONFIG_ROOT.glob("H[1-4]_*.yaml")):
        config = yaml.safe_load(path.read_text())
        fold = config["data"]["params"]["fold"]
        baseline_path = BASELINE_ROOT / f"F_weighted_parkinson_fold{fold}.yaml"
        baseline = yaml.safe_load(baseline_path.read_text())
        actual, reference = _scientific_parity(config, baseline)
        assert actual == reference
        if path.name.startswith("H2_"):
            assert config["loss"]["params"]["depression_pos_weight"] == expected_depression[fold]
