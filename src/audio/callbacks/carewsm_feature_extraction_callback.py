"""Persist packed extraction representations."""
import pickle
from pathlib import Path
from typing import Any
import numpy as np
from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS

class CAREWSMFeatureExtractionCallback(BaseCallback):
    def __init__(self, model_name: str, fold: int, care_root: str, wsm_root: str) -> None:
        self.model_name, self.fold = model_name, fold
        self.roots = {"CARE": Path(care_root), "WSM": Path(wsm_root)}
    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        for cached in trainer.cached_outputs.values():
            if cached.features is None or cached.sample_meta is None:
                continue
            features = cached.features.detach().cpu().numpy()
            logits = cached.preds.detach().cpu().numpy()
            if features.shape[1] != 192:
                raise ValueError("expected packed [batch, 192] extraction features")
            for values, prediction, meta in zip(features, logits, cached.sample_meta, strict=True):
                profile = str(meta["profile_id"]).removeprefix("PROFILE_")
                video = str(meta["video_id"]).removeprefix("VIDEO_")
                path = self.roots[meta["corpus"]] / "extracted_unimodal" / meta["label_id"] / meta["subgroup"] / f"PROFILE_{profile}" / f"VIDEO_{video}" / f"audio_{self.model_name}_fold{self.fold}.pickle"
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as handle:
                    pickle.dump({"shared_features": values[None,:64].astype(np.float32), "depression_features": values[None,64:128].astype(np.float32), "pd_features": values[None,128:].astype(np.float32), "logits": prediction.astype(np.float32)}, handle, protocol=pickle.HIGHEST_PROTOCOL)

@CALLBACKS.register("carewsm_feature_extraction_callback")
def carewsm_feature_extraction_callback(context: Any | None = None, **params: Any) -> CAREWSMFeatureExtractionCallback:
    del context
    return CAREWSMFeatureExtractionCallback(**params)
