#!/usr/bin/env python3
"""Compute validation-only group permutation importance for selected checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from chimera_ml.core.batch import Batch

import chimera_plugin  # noqa: F401

from audio.data.carewsm_audio_datamodule import CAREWSMAudioDataModule
from audio.models.carewsm_static_models import CAREWSMStaticGatedFusionModel

TEST_LOADERS = {"test_care", "test_wsm"}


def task_macro_recall(logits, targets, observed, task_index):
    mask = observed[:, task_index]
    if not bool(mask.any()):
        return float("nan")
    actual = targets[mask, task_index]
    predicted = (logits[mask, task_index] >= 0).to(torch.float32)
    recalls = []
    for label in (0.0, 1.0):
        support = actual == label
        if bool(support.any()):
            recalls.append(float((predicted[support] == label).float().mean()))
    return float(np.mean(recalls))


def evaluate(model, validation, permutation=None, generator=None):
    scores = {}
    for corpus in ("CARE", "WSM"):
        values, targets, observed = validation[corpus]
        values = values.clone()
        if permutation is not None:
            start, end = permutation
            order = torch.randperm(values.shape[0], generator=generator)
            values[:, 0, start:end] = values[order, 0, start:end]
        batch = Batch(
            inputs={"native": values},
            targets=targets,
            masks={"sequence_mask": torch.ones(values.shape[:2], dtype=torch.bool)},
        )
        logits = model(batch).preds.detach()
        for task_index, task in enumerate(("depression", "parkinson")):
            scores[f"{corpus.lower()}_{task}"] = task_macro_recall(
                logits, targets, observed, task_index
            )
    return scores


def select_validation_loaders(loaders):
    assert {"val_care", "val_wsm"}.issubset(loaders)
    selected_loaders = {name: loaders[name] for name in ("val_care", "val_wsm")}
    assert not (set(selected_loaders) & TEST_LOADERS)
    return selected_loaders


def collect_validation(data_module):
    selected_loaders = select_validation_loaders(data_module.val_dataloader())
    collected = {"CARE": [], "WSM": []}
    for loader_name in ("val_care", "val_wsm"):
        corpus = "CARE" if loader_name == "val_care" else "WSM"
        for batch in selected_loaders[loader_name]:
            observed = torch.stack(
                [item["observed_targets"] for item in batch.meta["sample_meta"]]
            )
            collected[corpus].append((batch.inputs["native"], batch.targets, observed))
    validation = {}
    for corpus, batches in collected.items():
        values, targets, observed = zip(*batches, strict=True)
        validation[corpus] = (
            torch.cat(values),
            torch.cat(targets),
            torch.cat(observed),
        )
    return validation


def feature_offsets(params, model_params):
    offsets = []
    offset = 0
    for family, dimension in zip(
        params["feature_family"], model_params["family_dims"], strict=True
    ):
        offsets.append((family, offset, offset + dimension))
        offset += dimension
    if offset != model_params["input_dim"]:
        raise ValueError("family dimensions do not cover the model input")
    indices = [index for _, start, end in offsets for index in range(start, end)]
    if len(set(indices)) != offset:
        raise ValueError("feature family offsets overlap")
    return offsets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    params = config["data"]["params"]
    data_module = CAREWSMAudioDataModule(**params)
    model_params = config["model"]["params"]
    model = CAREWSMStaticGatedFusionModel(**model_params)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(
        state["model_state_dict"] if "model_state_dict" in state else state
    )
    model.eval()
    validation = collect_validation(data_module)
    offsets = feature_offsets(params, model_params)
    baseline = evaluate(model, validation)
    rows = []
    for family, start, end in offsets:
        deltas = []
        for repeat in range(args.repeats):
            generator = torch.Generator().manual_seed(1000 + args.fold * 100 + repeat)
            permuted = evaluate(model, validation, (start, end), generator)
            deltas.append({key: baseline[key] - permuted[key] for key in baseline})
        for key in baseline:
            values = [delta[key] for delta in deltas]
            mean_delta = float(np.mean(values))
            rows.append(
                {
                    "fold": args.fold,
                    "family": family,
                    "corpus": key.split("_", 1)[0].upper(),
                    "task": key.split("_", 1)[1],
                    "val_baseline_macro_recall": baseline[key],
                    "val_permuted_macro_recall_mean": baseline[key] - mean_delta,
                    "val_permutation_delta_mean": mean_delta,
                    "val_permutation_delta_sample_sd": (
                        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    ),
                    "repeats": args.repeats,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"fold": args.fold, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
