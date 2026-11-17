#!/usr/bin/env python3
"""Compare one saved extraction per model/fold with CUDA-AMP checkpoint forward."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

import chimera_plugin
from chimera_ml.core.config import ExperimentConfig
from chimera_ml.training.builders import BuildContext, build_datamodule, build_model


PAIRS = {
    "id1": ("deep", "interp_B_mean_gated64_equal"),
    "id2": ("deep_phon", "interp_BP_mean_gated64_equal"),
    "id4": ("deep_egemaps_phon_artic", "interp_BEPA_mean_gated64_equal"),
}
LEDGER_PATH = Path("reports/audio_interpretable_feature_fusion/provenance_ledger.csv")
REPORT_PATH = Path("/tmp/carewsm-feature-audit-114/forward_amp.json")


def selected_checkpoints() -> dict[tuple[str, int], str]:
    with LEDGER_PATH.open(newline="", encoding="utf-8") as handle:
        return {
            (row["experiment_id"], int(row["fold"])): row["checkpoint_path"]
            for row in csv.DictReader(handle)
        }


def move_to_cuda(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.cuda(non_blocking=True)
    return value


def output_path(
    care_root: Path,
    sample_meta: dict[str, Any],
    model_id: str,
    fold: int,
) -> Path:
    profile = str(sample_meta["profile_id"]).removeprefix("PROFILE_")
    video = str(sample_meta["video_id"]).removeprefix("VIDEO_")
    return (
        care_root
        / "extracted_unimodal"
        / sample_meta["label_id"]
        / sample_meta["subgroup"]
        / f"PROFILE_{profile}"
        / f"VIDEO_{video}"
        / f"audio_{model_id}_fold{fold}.pickle"
    )


def compare_pair(
    model_id: str,
    config_prefix: str,
    experiment_id: str,
    fold: int,
    checkpoints: dict[tuple[str, int], str],
) -> dict[str, float | int | str]:
    config = ExperimentConfig.from_yaml(
        f"configs/audio_feature_extraction/{config_prefix}_fold{fold}.yaml"
    )
    context = BuildContext(config=config, stage="eval")
    datamodule = build_datamodule(config.section("data"), context=context)
    context.register(datamodule)
    model = build_model(config.section("model"), context=context)

    checkpoint = torch.load(
        checkpoints[(experiment_id, fold)], map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda().eval()

    batch = next(iter(datamodule.train_dataloader()["train_care"]))
    sample_meta = batch.meta["sample_meta"][0]
    batch.inputs = {name: move_to_cuda(value) for name, value in batch.inputs.items()}
    batch.targets = move_to_cuda(batch.targets)
    batch.masks = {name: move_to_cuda(value) for name, value in batch.masks.items()}

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
        output = model(batch)

    care_root = Path(config.section("data")["params"]["care_root"])
    with output_path(care_root, sample_meta, model_id, fold).open("rb") as handle:
        saved = pickle.load(handle)

    features = output.aux["features"][0].float().cpu().numpy()
    logits = output.preds[0].float().cpu().numpy()
    differences = {
        "shared": float(np.max(np.abs(saved["shared_features"] - features[None, :64]))),
        "depression": float(
            np.max(np.abs(saved["depression_features"] - features[None, 64:128]))
        ),
        "pd": float(np.max(np.abs(saved["pd_features"] - features[None, 128:]))),
        "logits": float(np.max(np.abs(saved["logits"] - logits))),
    }
    return {"model": model_id, "fold": fold, **differences}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce extraction AMP")

    chimera_plugin.register()
    checkpoints = selected_checkpoints()
    results = []
    for model_id, (config_prefix, experiment_id) in PAIRS.items():
        for fold in range(5):
            result = compare_pair(
                model_id, config_prefix, experiment_id, fold, checkpoints
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
