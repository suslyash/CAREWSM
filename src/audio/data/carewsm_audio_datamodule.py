"""Fold-aware joint CARE+WSM DataModule."""
from __future__ import annotations
import csv,json
from pathlib import Path
from typing import Any,Mapping
from chimera_ml.core.registry import DATAMODULES
from chimera_ml.data.datamodule import DataModule
from audio.data.carewsm_audio_dataset import CAREWSMAudioDataset,DataContractError

def _rows(path):
    with path.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def _label(v):
    if v is None or str(v).strip()=="":return None
    v=float(v)
    if v not in (0.,1.):raise DataContractError(f"non-binary label: {v}")
    return v
class CAREWSMAudioDataModule(DataModule):
    """Build joint fold TRAIN/VAL and separate fixed CARE/WSM TEST loaders."""
    def __init__(self,care_root,wsm_root,manifest,registry,fold=0,feature_family=None,feature_families=None,batch_size=32,num_workers=0,shuffle_train=False,max_sequence_length=None,max_train_samples=None,max_val_samples=None,max_test_samples=None):
        if fold not in range(5):raise ValueError("fold must be 0..4")
        if feature_families is not None and feature_family not in (None,"egemaps_lld"):raise ValueError("conflicting feature declarations")
        selected=tuple(feature_families or ([feature_family] if feature_family else []))
        if not selected or len(set(selected))!=len(selected):raise ValueError("feature families must be non-empty and non-duplicate")
        self.fold_index=fold;self.feature_families=selected;self.feature_family=selected[0] if len(selected)==1 else None;self.roots={"CARE":Path(care_root),"WSM":Path(wsm_root)}
        data=json.loads(Path(registry).read_text());self.feature_specs={k:v for k,v in data["sources"].items() if v.get("modality")=="audio"};self.records=self._manifest(Path(manifest))
        views=self._filter(self._views(),selected)
        def limit(name,rows):
            n=max_test_samples if name in ("care_test","wsm_test") else max_train_samples if name=="train" else max_val_samples if name=="val" else None
            if n is None:return rows
            if name in ("train","val"):return [r for c in ("CARE","WSM") for r in rows if r["corpus"]==c][:n]
            return rows[:n]
        ds={n:CAREWSMAudioDataset(limit(n,r),selected,self.roots,self.feature_specs,max_sequence_length) for n,r in views.items()}
        super().__init__(train_dataset=ds["train"],val_dataset={"joint":ds["val"]},test_dataset={"CARE":ds["care_test"],"WSM":ds["wsm_test"]},batch_size=batch_size,num_workers=num_workers,pin_memory=False,shuffle_train=shuffle_train);self.views=views
    def _manifest(self,path):
        out={}
        for row in _rows(path):
            key=(row["corpus"],row["sample_id"])
            if key in out:raise DataContractError(f"duplicate manifest key: {key}")
            row=dict(row)
            for t in ("depression","parkinson"):
                value=_label(row.get(t)); observed=value is not None; declared=str(row.get(t+"_observed",observed)).lower()=="true"
                if declared!=observed:raise DataContractError(f"observed-label mismatch: {key} {t}")
                row[t]=value;row[t+"_observed"]=observed
            out[key]=row
        return out
    def _fold(self,corpus,part):
        out=[]
        for row in _rows(self.roots[corpus]/"splits"/"folds"/f"fold_{self.fold_index}_{part}.csv"):
            key=(corpus,row["sample_id"])
            if key not in self.records:raise DataContractError(f"fold row absent from manifest: {key}")
            r=dict(self.records[key])
            if row.get("profile_id") and row["profile_id"]!=r.get("profile_id"):raise DataContractError(f"profile mismatch: {key}")
            out.append(r)
        return out
    def _views(self):
        train=self._fold("CARE","train")+self._fold("WSM","train");val=self._fold("CARE","val")+self._fold("WSM","val");tests={}
        for c in self.roots:tests[c]=[self.records[(c,r["sample_id"])] for r in sorted(_rows(self.roots[c]/"splits"/"test.csv"),key=lambda x:x["sample_id"])]
        self._isolated(train,val);self._isolated(train+val,tests["CARE"]+tests["WSM"]);return {"train":train,"val":val,"care_test":tests["CARE"],"wsm_test":tests["WSM"]}
    @staticmethod
    def _isolated(a,b):
        keys={(r["corpus"],r["sample_id"]) for r in a};other={(r["corpus"],r["sample_id"]) for r in b};profiles={(r["corpus"],r["profile_id"]) for r in a};other_profiles={(r["corpus"],r["profile_id"]) for r in b}
        if keys&other or profiles&other_profiles:raise DataContractError("sample/profile leakage between populations")
    @staticmethod
    def _filter(views,selected):return {n:[r for r in rows if all(r.get(f+"_status")=="VALID" for f in selected)] for n,rows in views.items()}
    def describe_context(self,context):
        context.set("data.module",self);context.set("data.fold",self.fold_index);context.set("data.feature_family",self.feature_family or list(self.feature_families))
        for n,r in self.views.items():context.set("data."+n+"_count",len(r))
@DATAMODULES.register("carewsm_joint")
def carewsm_joint(context:Any|None=None,**params):return CAREWSMAudioDataModule(**params)
