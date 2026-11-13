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
from torch.utils.data import DataLoader

import chimera_plugin  # noqa: F401
from chimera_ml.core.batch import Batch

from audio.data.carewsm_audio_datamodule import CAREWSMAudioDataModule
from audio.models.carewsm_static_models import CAREWSMStaticGatedFusionModel

FAMILIES = (
    "emotionmsp_embeds", "emotionmsp_trajectories", "trillsson4_1",
    "wav2vec_xlsr300m", "egemaps_func", "disvoice_phonation_static",
    "disvoice_articulation_static",
)

def macro_recall(logits: torch.Tensor, targets: torch.Tensor, observed: torch.Tensor) -> float:
    values = []
    predictions = (logits >= 0).to(torch.float32)
    for task in range(2):
        mask = observed[:, task]
        if not bool(mask.any()):
            continue
        actual = targets[mask, task]
        predicted = predictions[mask, task]
        recalls = []
        for label in (0.0, 1.0):
            support = actual == label
            if bool(support.any()):
                recalls.append(float((predicted[support] == label).float().mean()))
        if recalls:
            values.append(float(np.mean(recalls)))
    return float(np.mean(values))

def evaluate(model, batches, permutation: tuple[int, int] | None) -> float:
    logits = []
    targets = []
    observed = []
    for inputs, batch_targets, batch_observed, corpus in batches:
        values = inputs.clone()
        if permutation is not None:
            start, end = permutation
            for name in ("CARE", "WSM"):
                indexes = [i for i, value in enumerate(corpus) if value == name]
                if len(indexes) > 1:
                    order = torch.tensor(indexes)[torch.randperm(len(indexes))]
                    values[indexes, 0, start:end] = values[order, 0, start:end]
        model_batch = Batch(
            inputs={"native": values},
            targets=batch_targets,
            masks={"sequence_mask": torch.ones(
                values.shape[:2], dtype=torch.bool
            )},
        )
        output = model(model_batch)
        logits.append(output.preds.detach().cpu())
        targets.append(batch_targets)
        observed.append(batch_observed)
    return macro_recall(torch.cat(logits), torch.cat(targets), torch.cat(observed))

def collect_batches(data_module) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]:
    result = []
    loaders = data_module.val_dataloader()
    for name, loader in loaders.items():
        for batch in loader:
            values = batch.inputs["native"]
            observed = torch.stack(
                [item["observed_targets"] for item in batch.meta["sample_meta"]]
            )
            corpus = name.split("_")[1].upper()
            result.append((values, batch.targets, observed, [corpus] * values.shape[0]))
    return result

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
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()
    batches = collect_batches(data_module)
    offsets = []
    offset = 0
    for family, dimension in zip(params["feature_family"], model_params["family_dims"], strict=True):
        offsets.append((family, offset, offset + dimension))
        offset += dimension
    baseline = evaluate(model, batches, None)
    rows = []
    for family, start, end in offsets:
        deltas = []
        for repeat in range(args.repeats):
            torch.manual_seed(1000 + args.fold * 100 + repeat)
            permuted = evaluate(model, batches, (start, end))
            deltas.append(baseline - permuted)
        rows.append({"fold": args.fold, "family": family, "val_baseline_macro_recall": baseline, "val_permuted_macro_recall_mean": float(np.mean([baseline - delta for delta in deltas])), "val_permutation_delta_mean": float(np.mean(deltas)), "val_permutation_delta_sample_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0, "repeats": args.repeats})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"fold": args.fold, "rows": len(rows), "baseline_val_macro_recall": baseline}, indent=2))

if __name__ == "__main__":
    main()
