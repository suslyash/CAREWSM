from __future__ import annotations
from typing import Any
import torch
from chimera_ml.core.registry import METRICS
from chimera_ml.metrics.base import BaseMetric

class BinaryMetric(BaseMetric):
    def __init__(self, name: str, task: int = 0) -> None:
        self.name, self.task = name, task
        self.reset()
    def reset(self) -> None:
        self.predictions, self.targets, self.observed = [], [], []
    def update(self, output: Any, batch: Any) -> None:
        mask = batch.get_masks("observed_targets")
        if mask is None and batch.meta and "sample_meta" in batch.meta:
            mask = torch.stack([item["observed_targets"] for item in batch.meta["sample_meta"]])
        if mask is None: raise ValueError("Batch does not contain observed-label masks")
        self.predictions.append((output.preds > 0).detach().cpu()); self.targets.append(batch.targets.detach().cpu()); self.observed.append(mask.detach().cpu())
    def compute(self) -> dict[str, float]:
        if not self.predictions: return {self.name: 0.0, "count": 0.0}
        p=torch.cat(self.predictions)[:,self.task].int(); y=torch.cat(self.targets)[:,self.task].int(); m=torch.cat(self.observed)[:,self.task].bool(); p,y=p[m],y[m]
        tp=int(((p==1)&(y==1)).sum()); tn=int(((p==0)&(y==0)).sum()); fp=int(((p==1)&(y==0)).sum()); fn=int(((p==0)&(y==1)).sum()); n=len(y)
        pr=tp/max(tp+fn,1); hr=tn/max(tn+fp,1); pf=2*tp/max(2*tp+fp+fn,1); hf=2*tn/max(2*tn+fp+fn,1)
        values={"Macro F1":(pf+hf)/2,"Weighted F1":(pf*(tp+fn)+hf*(tn+fp))/max(n,1),"Macro Recall":(pr+hr)/2,"Weighted Recall":(pr*(tp+fn)+hr*(tn+fp))/max(n,1),"Patient Recall":pr,"Healthy Recall":hr}
        return {self.name: values[self.name], "count": float(n)}

@METRICS.register("carewsm_macro_f1")
def carewsm_macro_f1(context: Any | None = None, **params: Any) -> BinaryMetric: del context; return BinaryMetric("Macro F1", **params)
@METRICS.register("carewsm_weighted_f1")
def carewsm_weighted_f1(context: Any | None = None, **params: Any) -> BinaryMetric: del context; return BinaryMetric("Weighted F1", **params)
@METRICS.register("carewsm_macro_recall")
def carewsm_macro_recall(context: Any | None = None, **params: Any) -> BinaryMetric: del context; return BinaryMetric("Macro Recall", **params)
@METRICS.register("carewsm_weighted_recall")
def carewsm_weighted_recall(context: Any | None = None, **params: Any) -> BinaryMetric: del context; return BinaryMetric("Weighted Recall", **params)
@METRICS.register("carewsm_patient_recall")
def carewsm_patient_recall(context: Any | None = None, **params: Any) -> BinaryMetric: del context; return BinaryMetric("Patient Recall", **params)
@METRICS.register("carewsm_healthy_recall")
def carewsm_healthy_recall(context: Any | None = None, **params: Any) -> BinaryMetric: del context; return BinaryMetric("Healthy Recall", **params)


def corpus_average(care: dict[str, float], wsm: dict[str, float]) -> dict[str, float]:
    return {name: (care[name] + wsm[name]) / 2 for name in care}
