"""Persist fold-specific recording representations from cached evaluation outputs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


class CAREWSMFeatureExtractionCallback(BaseCallback):
    """Write new recording pickles without replacing existing extraction results."""

    def __init__(
        self,
        model_name: str,
        fold: int,
        care_root: str,
        wsm_root: str,
        report_path: str,
    ) -> None:
        self.model_name = model_name
        self.fold = fold
        self.roots = {"CARE": Path(care_root), "WSM": Path(wsm_root)}
        self.report_path = Path(report_path)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        del epoch, logs
        report: dict[str, Any] = {
            "model_name": self.model_name,
            "fold": self.fold,
            "created": 0,
            "existing": 0,
            "skipped": 0,
            "failed": [],
            "loaders": {},
        }
        for split, cached in trainer.cached_outputs.items():
            counts = {"created": 0, "existing": 0, "skipped": 0, "failed": 0}
            report["loaders"][split] = counts
            if cached.features is None or cached.sample_meta is None:
                counts["skipped"] += 1
                report["skipped"] += 1
                continue
            features = cached.features.detach().cpu().numpy()
            logits = cached.preds.detach().cpu().numpy()
            if features.shape[1] != 192:
                raise ValueError("expected packed [batch, 192] extraction features")
            for values, prediction, meta in zip(
                features, logits, cached.sample_meta, strict=True
            ):
                prediction = prediction.astype(np.float32)
                profile = str(meta["profile_id"]).removeprefix("PROFILE_")
                video = str(meta["video_id"]).removeprefix("VIDEO_")
                path = (
                    self.roots[meta["corpus"]]
                    / "extracted_unimodal"
                    / meta["label_id"]
                    / meta["subgroup"]
                    / f"PROFILE_{profile}"
                    / f"VIDEO_{video}"
                    / f"audio_{self.model_name}_fold{self.fold}.pickle"
                )
                if path.exists():
                    counts["existing"] += 1
                    report["existing"] += 1
                    continue
                payload = {
                    "shared_features": values[None, :64].astype(np.float32),
                    "depression_features": values[None, 64:128].astype(np.float32),
                    "pd_features": values[None, 128:].astype(np.float32),
                    "logits": prediction,
                    "probs": (1.0 / (1.0 + np.exp(-prediction))).astype(np.float32),
                }
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as handle:
                        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except FileExistsError:
                    counts["existing"] += 1
                    report["existing"] += 1
                except (OSError, pickle.PickleError) as exc:
                    counts["failed"] += 1
                    report["failed"].append({"path": str(path), "error": str(exc)})
                else:
                    counts["created"] += 1
                    report["created"] += 1
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)


@CALLBACKS.register("carewsm_feature_extraction_callback")
def carewsm_feature_extraction_callback(
    context: Any | None = None, **params: Any
) -> CAREWSMFeatureExtractionCallback:
    del context
    return CAREWSMFeatureExtractionCallback(**params)
