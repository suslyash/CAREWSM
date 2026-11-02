"""Focused TASK-004 registry and real-batch smoke."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
import chimera_plugin
from chimera_ml.core.registry import DATAMODULES, LOSSES, METRICS, MODELS


def observed_mask(batch):
    mask = batch.get_masks("observed_targets")
    if mask is not None:
        return mask
    return torch.stack([item["observed_targets"] for item in batch.meta["sample_meta"]])


def main() -> None:
    chimera_plugin.register()
    data = DATAMODULES.create(
        "carewsm_joint", care_root="/media/maxim/Databases/CARE",
        wsm_root="/media/maxim/Databases/WSM_NEW",
        manifest="reports/feature_audit/sample_feature_availability.csv",
        registry="/media/maxim/Databases/CARE/splits/features.json",
        fold=0, feature_family="egemaps_lld", batch_size=2,
    )
    batch = next(iter(data.train_dataloader()))
    batch.inputs["native"] = batch.inputs["native"][:, :32]
    batch.masks["sequence_mask"] = batch.masks["sequence_mask"][:, :32]
    batch.masks["observed_targets"] = observed_mask(batch)
    input_dim = batch.inputs["native"].shape[-1]
    for name, params in (("carewsm_mptsnet", {}), ("carewsm_md_former", {"num_heads": 2})):
        model = MODELS.create(name, input_dim=input_dim, hidden_dim=8, **params)
        output = model(batch)
        loss = LOSSES.create("carewsm_masked_bce")(output, batch)
        loss.backward()
        assert torch.isfinite(output.preds).all()
        print(name, tuple(output.preds.shape), "finite")
    for name in ("carewsm_macro_f1", "carewsm_weighted_f1", "carewsm_macro_recall",
                 "carewsm_weighted_recall", "carewsm_patient_recall", "carewsm_healthy_recall"):
        metric = METRICS.create(name)
        metric.update(output, batch)
        print(name, metric.compute())
    try:
        MODELS.create("carewsm_tscmamba", input_dim=input_dim, hidden_dim=8)
    except ModuleNotFoundError as exc:
        print("carewsm_tscmamba BLOCKED:", exc)


if __name__ == "__main__":
    main()
