"""Direct native-feature Dataset for CAREWSM audio."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from torch.utils.data import Dataset

class DataContractError(ValueError):
    """A manifest, split, or feature violates the data contract."""

def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def _load(path: Path, spec: Mapping[str, Any], family: str) -> np.ndarray:
    if spec.get("format") == "csv":
        rows = _rows(path)
        if not rows: raise DataContractError(f"empty feature artifact: {path}")
        cols = [c for c in rows[0] if c not in set(spec.get("metadata", []))]
        try: values = np.asarray([[float(row[c]) for c in cols] for row in rows], dtype=np.float32)
        except (KeyError, ValueError) as e: raise DataContractError(f"invalid CSV schema: {path}") from e
    else:
        try:
            with np.load(path, allow_pickle=False) as z:
                keys = spec.get("embedding_keys", [])
                key = "embed" if family == "emotionmsp_embeds" else (keys[0] if keys else None)
                if key is None or key not in z.files: raise DataContractError(f"missing NPZ key {key!r}: {path}")
                values = np.asarray(z[key], dtype=np.float32)
        except (OSError, ValueError) as e: raise DataContractError(f"unreadable feature artifact: {path}") from e
    if values.ndim == 1: values = values.reshape(1, -1)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise DataContractError(f"invalid feature array: {path} {values.shape}")
    return values

class CAREWSMAudioDataset(Dataset[dict[str, Any]]):
    """Load one native family or ordered static families for selected records."""
    def __init__(self, records, feature_families, roots, feature_specs, max_sequence_length=None):
        self.records=tuple(records); self.feature_families=tuple(feature_families)
        self.roots={k:Path(v) for k,v in roots.items()}; self.feature_specs=feature_specs
        self.max_sequence_length=max_sequence_length
        if not self.feature_families or len(set(self.feature_families)) != len(self.feature_families): raise ValueError("feature families must be non-empty and non-duplicate")
        unknown=set(self.feature_families)-set(feature_specs)
        if unknown: raise ValueError(f"unknown feature family: {sorted(unknown)}")
        if max_sequence_length is not None and max_sequence_length < 1: raise ValueError("max_sequence_length must be positive")
    def __len__(self): return len(self.records)
    def _path(self, r, family):
        root=self.roots[r["corpus"]]; profile=str(r.get("profile_id","")); video=str(r.get("video_id",r["sample_id"]))
        profile=profile if profile.startswith("PROFILE_") else "PROFILE_"+profile; video=video if video.startswith("VIDEO_") else "VIDEO_"+video
        return root/"extracted"/str(r.get("label_id",""))/str(r.get("subgroup",""))/profile/video/self.feature_specs[family]["file"]
    def __getitem__(self, index):
        r=self.records[index]; loaded=[]
        for family in self.feature_families:
            if r.get(f"{family}_status",r.get(f"{family}_valid")) not in ("VALID",True): raise DataContractError(f"feature is not VALID: {r['sample_id']} {family}")
            path=self._path(r,family); loaded.append((_load(path,self.feature_specs[family],family),path))
        arrays=[a for a,_ in loaded]
        if len(arrays)>1:
            if any(a.shape[0]!=1 for a in arrays): raise DataContractError("multiple families require static T=1 arrays; temporal alignment is undefined")
            values=np.concatenate(arrays,axis=1)
        else:
            values=arrays[0]
            if self.max_sequence_length and len(values)>self.max_sequence_length:
                edges=np.linspace(0,len(values),self.max_sequence_length+1,dtype=np.int64)
                values=np.stack([values[edges[i]:edges[i+1]].mean(axis=0) for i in range(self.max_sequence_length)])
        observed=torch.tensor([bool(r.get("depression_observed")),bool(r.get("parkinson_observed"))],dtype=torch.bool)
        targets=torch.tensor([float(r.get("depression",0)) if observed[0] else 0.,float(r.get("parkinson",0)) if observed[1] else 0.],dtype=torch.float32)
        tensor=torch.from_numpy(values.astype(np.float32,copy=False))
        return {"inputs":{"native":tensor},"targets":targets,"meta":{"corpus":r["corpus"],"sample_id":r["sample_id"],"profile_id":r["profile_id"],"feature_families":list(self.feature_families),"feature_dimensions":[int(a.shape[1]) for a in arrays],"feature_paths":[str(p) for _,p in loaded],"observed_targets":observed,"masks":{"sequence_mask":torch.ones(tensor.shape[0],dtype=torch.bool)}}}
