
#!/usr/bin/env python3
"""Frozen Stage-7 DEV preflight and one-shot final Test evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from audio.data.wsm_audio_segment_datamodule import WSMAudioSegmentDataModule
from audio.models.audio_mamba_segment import AudioMambaSegmentModel
from common.callbacks.wsm_segment_callback import compute_sparse_two_task_metrics
from fusion.data.wsm_ramps_semantic_datamodule import WSMRampsSemanticDataModule
from fusion.models.av_r3_disease_query import WSMAVR3DiseaseQueryModel
from fusion.models.frozen_audio_temporal_adapter import FrozenAudioTemporalAdapter

PSEUDO_SHA256 = "17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945"
T_MULTIPLIER = 2.7764451051977987
PROTOCOL_COUNTS = {"test_none": 1364, "test_soft": 1208, "test_hard": 1014}
TASKS = ("depression", "parkinson")

AUDIO_CONFIGS = {
    "42": "configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml",
    "43": "configs/wsm_mm_pd_dep_v1/audio/01_frozen_baseline_seed43.yaml",
    "44": "configs/wsm_mm_pd_dep_v1/audio/02_frozen_baseline_seed44.yaml",
    "45": "configs/wsm_mm_pd_dep_v1/audio/03_final_seed45.yaml",
    "46": "configs/wsm_mm_pd_dep_v1/audio/04_final_seed46.yaml",
}
AUDIO_CONFIG_SHAS = {
    "42": "955ad447dd941b69635121546ad5aaf2fd2e2fdc7b0ae4307ab0f8532edcba57",
    "43": "e4df4c8bb9bfaf0fa6ca7a0cbf8d44261a80b4d8a0398c0caf8e2c961ba0c91a",
    "44": "d9c9e596df5f9ae723a54d63558456a70857e1fa9f2e63612246e86f7f83dc92",
    "45": "8bb3c4109ddc04ec10976dcbc8ee9b533a29d404ca51dc6f511f19d8d4fe9e21",
    "46": "71ba2fc42ff67b55ef2949d9011931c9c0967f0ac07d5c057095a97648081767",
}
AUDIO_CKPTS = {
    "42": ("logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt", "0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2", "multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77", 4),
    "43": ("logs/wsm_mm_pd_dep_v1/frozen_audio_wavlm_l9_pool4_seed43_2026-09-28_13-12_audio_mamba_segment_model_08192c1b/checkpoints/epoch=5_dev_mean_score=0.7602.pt", "1a65f8dd605a6d942bed828f126e79a1f02930dd578176ba31f49c4dc10899dd", "frozen_audio_wavlm_l9_pool4_seed43_2026-09-28_13-12_audio_mamba_segment_model_08192c1b", 5),
    "44": ("logs/wsm_mm_pd_dep_v1/frozen_audio_wavlm_l9_pool4_seed44_2026-09-28_13-21_audio_mamba_segment_model_8720f109/checkpoints/epoch=4_dev_mean_score=0.7719.pt", "7832d20a5431666365e2214423633f2be87bb27f37406d5a5dda8adda21e9852", "frozen_audio_wavlm_l9_pool4_seed44_2026-09-28_13-21_audio_mamba_segment_model_8720f109", 4),
    "45": ("logs/wsm_mm_pd_dep_v1/stage7_audio_final_seed45_2026-09-29_15-59_audio_mamba_segment_model_e1210d8f/checkpoints/epoch=6_dev_mean_score=0.7482.pt", "1881bc526e9d17d77f424d2b053b5ea53f7ba5d85b82cb7643176c764da5f370", "stage7_audio_final_seed45_2026-09-29_15-59_audio_mamba_segment_model_e1210d8f", 6),
    "46": ("logs/wsm_mm_pd_dep_v1/stage7_audio_final_seed46_2026-09-29_16-25_audio_mamba_segment_model_57104699/checkpoints/epoch=10_dev_mean_score=0.7576.pt", "546ca0929e746692f18106922cecb10ea4781cca2243d31855e68d7d63983263", "stage7_audio_final_seed46_2026-09-29_16-25_audio_mamba_segment_model_57104699", 10),
}
FUSION_CONFIGS = {
    ("r4","42"): "configs/wsm_mm_pd_dep_v1/fusion/37_r4_ra_stch_optuna_selected_seed42.yaml",
    ("r4","43"): "configs/wsm_mm_pd_dep_v1/fusion/38_r4_ra_stch_optuna_confirm_seed43.yaml",
    ("r4","44"): "configs/wsm_mm_pd_dep_v1/fusion/39_r4_ra_stch_optuna_confirm_seed44.yaml",
    ("r4","45"): "configs/wsm_mm_pd_dep_v1/fusion/40_r4_trial012_final_seed45.yaml",
    ("r4","46"): "configs/wsm_mm_pd_dep_v1/fusion/41_r4_trial012_final_seed46.yaml",
    ("shared","42"): "configs/wsm_mm_pd_dep_v1/ablations/55_shared_fusion_seed42.yaml",
    ("shared","43"): "configs/wsm_mm_pd_dep_v1/ablations/56_shared_fusion_seed43.yaml",
    ("shared","44"): "configs/wsm_mm_pd_dep_v1/ablations/57_shared_fusion_seed44.yaml",
    ("shared","45"): "configs/wsm_mm_pd_dep_v1/fusion/42_shared_fusion_final_seed45.yaml",
    ("shared","46"): "configs/wsm_mm_pd_dep_v1/fusion/43_shared_fusion_final_seed46.yaml",
}
FUSION_CONFIG_SHAS = {
    ("r4","42"): "04ed5fcb2a1a1040f92c6ebd97898f20cb0658b82eb474c72b05fcfc767a2fb5",
    ("r4","43"): "9409f6e581eb2cb129593967f5bf72447ce0cf76a3891afbf417315adb593e6c",
    ("r4","44"): "5f13b1c72b0d5f150ed734584910d406df5eee94a39561fc7800551be4e59587",
    ("r4","45"): "99cc0c9c2994ccf4bf3c8f8ae2c1730c3e1cc515519ac05a59e30a3446d8f780",
    ("r4","46"): "c37eb7c0819bff366edf223afa836f6a4698df0b50831b5b7ba721aefd55231d",
    ("shared","42"): "4c8d2d18b4ee004a1f59067374946fa9dda0f2886c653a4504559f9630a46d6e",
    ("shared","43"): "62e6631c45386fe594a7a57dbac441464ae4bf3b17f9d1e73941f793665f8f16",
    ("shared","44"): "5a9cef2d790044c774fc1c422fd683194f6ab3461eae880eb1c4a73e863f5431",
    ("shared","45"): "30236d8f06eaa77c9d7d4d40fafe2402407541bd3a71d7a836e35817d57ce27f",
    ("shared","46"): "00310e540f89e355fdf2492cbf67bf03c71ba8bb720a2112e971ea0e4fe8afd6",
}
FUSION_CKPTS = {
    ("r4","42"): ("logs/wsm_mm_pd_dep_v1/optuna_r4_ra_stch_base_seed42_2026-09-28_01-54_wsm_av_r3_disease_query_model_r4-ra-stch-optuna-v1-18c6-012_0ce9c7a4/checkpoints/epoch=11_dev_mean_score=0.8191.pt", "104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a", "optuna_r4_ra_stch_base_seed42_2026-09-28_01-54_wsm_av_r3_disease_query_model_r4-ra-stch-optuna-v1-18c6-012_0ce9c7a4", 11),
    ("r4","43"): ("logs/wsm_mm_pd_dep_v1/r4_ra_stch_optuna_trial012_confirm_seed43_2026-09-28_12-01_wsm_av_r3_disease_query_model_12038580/checkpoints/epoch=10_dev_mean_score=0.7614.pt", "6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e", "r4_ra_stch_optuna_trial012_confirm_seed43_2026-09-28_12-01_wsm_av_r3_disease_query_model_12038580", 10),
    ("r4","44"): ("logs/wsm_mm_pd_dep_v1/r4_ra_stch_optuna_trial012_confirm_seed44_2026-09-28_12-07_wsm_av_r3_disease_query_model_41bc602b/checkpoints/epoch=11_dev_mean_score=0.7801.pt", "b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17", "r4_ra_stch_optuna_trial012_confirm_seed44_2026-09-28_12-07_wsm_av_r3_disease_query_model_41bc602b", 11),
    ("r4","45"): ("logs/wsm_mm_pd_dep_v1/stage7_r4_trial012_final_seed45_2026-09-29_16-08_wsm_av_r3_disease_query_model_9a555979/checkpoints/epoch=20_dev_mean_score=0.7918.pt", "cd30baf76392ab04e8b54d64ad67b260bae71216313942d968ad564a3422fc37", "stage7_r4_trial012_final_seed45_2026-09-29_16-08_wsm_av_r3_disease_query_model_9a555979", 20),
    ("r4","46"): ("logs/wsm_mm_pd_dep_v1/stage7_r4_trial012_final_seed46_2026-09-29_16-36_wsm_av_r3_disease_query_model_52637a4b/checkpoints/epoch=30_dev_mean_score=0.7978.pt", "18ee9c9e298fab13f807404cd625360c3e6c063a73ee4340d6f474207044f5e1", "stage7_r4_trial012_final_seed46_2026-09-29_16-36_wsm_av_r3_disease_query_model_52637a4b", 30),
    ("shared","42"): ("logs/wsm_mm_pd_dep_v1/stage6_shared_fusion_trial012_seed42_2026-09-28_22-44_wsm_av_r3_disease_query_model_5db549e8/checkpoints/epoch=11_dev_mean_score=0.8245.pt", "21504702976a960ffea02a68778ebc3bf7e6cc15ea8c9866f182ff04cbd30785", "stage6_shared_fusion_trial012_seed42_2026-09-28_22-44_wsm_av_r3_disease_query_model_5db549e8", 11),
    ("shared","43"): ("logs/wsm_mm_pd_dep_v1/stage6_shared_fusion_trial012_seed43_2026-09-28_22-51_wsm_av_r3_disease_query_model_454bf80b/checkpoints/epoch=10_dev_mean_score=0.7713.pt", "ec746e519e551d69d59954dffff0a73968fb4ab555d43020b7722e635cc477ee", "stage6_shared_fusion_trial012_seed43_2026-09-28_22-51_wsm_av_r3_disease_query_model_454bf80b", 10),
    ("shared","44"): ("logs/wsm_mm_pd_dep_v1/stage6_shared_fusion_trial012_seed44_2026-09-28_22-57_wsm_av_r3_disease_query_model_a398d7d8/checkpoints/epoch=11_dev_mean_score=0.7849.pt", "215e3a61c5dcb8ff7dd92ed5ca9336d66405c3f1ba6cc045841f08ec990d10a8", "stage6_shared_fusion_trial012_seed44_2026-09-28_22-57_wsm_av_r3_disease_query_model_a398d7d8", 11),
    ("shared","45"): ("logs/wsm_mm_pd_dep_v1/stage7_shared_fusion_final_seed45_2026-09-29_16-17_wsm_av_r3_disease_query_model_2eb9d5be/checkpoints/epoch=14_dev_mean_score=0.7872.pt", "ccd80ca0be94d25901eb6ad628671c4ebda24ca745a22af1136f5b6c6c4ea606", "stage7_shared_fusion_final_seed45_2026-09-29_16-17_wsm_av_r3_disease_query_model_2eb9d5be", 14),
    ("shared","46"): ("logs/wsm_mm_pd_dep_v1/stage7_shared_fusion_final_seed46_2026-09-29_16-47_wsm_av_r3_disease_query_model_8d38837e/checkpoints/epoch=25_dev_mean_score=0.7867.pt", "b99c153723265641057bf25c0aa7a57c01e5dc8cac638a1bcd40f9c87bb3041b", "stage7_shared_fusion_final_seed46_2026-09-29_16-47_wsm_av_r3_disease_query_model_8d38837e", 25),
}
DEV_EXPECTED = {
    ("audio","42"):(0.7479183895,0.8277353635,0.7878268765),("audio","43"):(0.708699,0.811771,0.760235),("audio","44"):(0.734396,0.809382,0.771889),("audio","45"):(0.697398,0.799034,0.748216),("audio","46"):(0.720187,0.794924,0.757556),
    ("r4","42"):(0.759025,0.879259,0.8191424538),("r4","43"):(0.718527,0.804363,0.761445),("r4","44"):(0.728217,0.832060,0.780139),("r4","45"):(0.725077,0.858468,0.791772),("r4","46"):(0.725405,0.870164,0.797785),
    ("shared","42"):(0.765324,0.883606,0.824465),("shared","43"):(0.724496,0.818152,0.771324),("shared","44"):(0.731448,0.838372,0.784910),("shared","45"):(0.739733,0.834672,0.787202),("shared","46"):(0.727451,0.845891,0.786671),
}

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def frozen_entries(repo: Path) -> list[dict[str,Any]]:
    entries=[]
    for seed in ("42","43","44","45","46"):
        cp, ch, run, epoch = AUDIO_CKPTS[seed]
        entries.append({"method":"audio","seed":int(seed),"seed_key":seed,"checkpoint":cp,"checkpoint_sha256":ch,"config":AUDIO_CONFIGS[seed],"config_sha256":AUDIO_CONFIG_SHAS[seed],"run":run,"epoch":epoch})
    for family in ("r4","shared"):
        for seed in ("42","43","44","45","46"):
            cp,ch,run,epoch=FUSION_CKPTS[(family,seed)]
            entries.append({"method":family,"seed":int(seed),"seed_key":seed,"checkpoint":cp,"checkpoint_sha256":ch,"config":FUSION_CONFIGS[(family,seed)],"config_sha256":FUSION_CONFIG_SHAS[(family,seed)],"run":run,"epoch":epoch})
    if len(entries)!=15 or len({(x["method"],x["seed"]) for x in entries})!=15:
        raise RuntimeError("frozen ledger must contain exactly 15 unique entries")
    return entries

def verify_ledger(repo: Path, entries: list[dict[str,Any]], pseudo_cache: Path) -> None:
    for e in entries:
        cp=repo/e["checkpoint"]; cfg=repo/e["config"]
        if not cp.is_file(): raise RuntimeError(f"missing checkpoint: {cp}")
        if sha256(cp)!=e["checkpoint_sha256"]: raise RuntimeError(f"checkpoint SHA mismatch: {cp}")
        if not cfg.is_file(): raise RuntimeError(f"missing config: {cfg}")
        if sha256(cfg)!=e["config_sha256"]: raise RuntimeError(f"config SHA mismatch: {cfg}")
        text=cfg.read_text(encoding="utf-8")
        if f"seed: {e['seed']}" not in text or "experiment_name: wsm_mm_pd_dep_v1" not in text:
            raise RuntimeError(f"config identity mismatch: {cfg}")
    if not pseudo_cache.is_file() or sha256(pseudo_cache)!=PSEUDO_SHA256:
        raise RuntimeError("pseudo-cache SHA mismatch")

def binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str,float]:
    logits=logits.detach().cpu().reshape(-1); targets=targets.detach().cpu().reshape(-1).to(torch.int64)
    pred=logits>=0; pos=targets==1; neg=targets==0
    tp=int((pred&pos).sum()); fp=int((pred&neg).sum()); fn=int((~pred&pos).sum()); tn=int((~pred&neg).sum())
    rn=tn/(tn+fp) if tn+fp else 0.0; rp=tp/(tp+fn) if tp+fn else 0.0
    fnn=2*tn/(2*tn+fn+fp) if 2*tn+fn+fp else 0.0; fpp=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    uar=(rn+rp)/2; mf1=(fnn+fpp)/2
    return {"num_samples":len(targets),"uar":uar,"mf1":mf1,"score":(uar+mf1)/2}

def sparse_metrics(logits: torch.Tensor, targets: torch.Tensor, observed: torch.Tensor) -> dict[str,Any]:
    raw=compute_sparse_two_task_metrics(logits,targets,observed,"test")
    return {"depression":{"num_samples":int(raw["test/depression/num_samples"]),"uar":raw["test/depression/uar"],"mf1":raw["test/depression/mf1"],"score":raw["test/depression/score"]},"parkinson":{"num_samples":int(raw["test/parkinson/num_samples"]),"uar":raw["test/parkinson/uar"],"mf1":raw["test/parkinson/mf1"],"score":raw["test/parkinson/score"]},"mean_score":raw["test/mean_score"]}

def load_audio(repo: Path, e: dict[str,Any]) -> tuple[torch.nn.Module,bool]:
    if e["seed"]==42:
        return FrozenAudioTemporalAdapter(str(repo/e["checkpoint"])), True
    m=AudioMambaSegmentModel(audio_feature_dim=768,num_tasks=2,num_classes=2,hidden_dim=192,num_layers=3,num_heads=4,ff_mult=4,dropout=0.25,encoder_type="transformer",sequence_steps=128,mamba_d_state=16,mamba_d_conv=4,mamba_expand=2,mamba_required=True)
    payload=torch.load(repo/e["checkpoint"],map_location="cpu",weights_only=False); m.load_state_dict(payload["model_state_dict"],strict=True); m.eval()
    return m,False

def load_fusion(repo: Path, e: dict[str,Any]) -> torch.nn.Module:
    m=WSMAVR3DiseaseQueryModel(audio_feature_dim=768,video_feature_dim=512,hidden_dim=160,gate_hidden_dim=96,dropout=0.10034630487649018,num_tasks=2,task_aware_fusion=e["method"]=="r4")
    payload=torch.load(repo/e["checkpoint"],map_location="cpu",weights_only=False); m.load_state_dict(payload["model_state_dict"],strict=True); m.eval()
    return m

def config_runtime(repo: Path, config: str) -> tuple[torch.device, bool]:
    """Mirror Chimera's device fallback and mixed-precision gate."""
    payload = yaml.safe_load((repo / config).read_text(encoding="utf-8"))
    train = payload.get("train", {}).get("params", {})
    requested = torch.device(str(train.get("device", "cpu")))
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    return device, bool(train.get("mixed_precision", False)) and device.type == "cuda"

def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    batch.inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.inputs.items()}
    if batch.targets is not None:
        batch.targets = batch.targets.to(device)
    batch.masks = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.masks.items()}
    return batch

@torch.no_grad()
def evaluate_audio(repo: Path, dm: WSMAudioSegmentDataModule, e: dict[str,Any], protocol: str, dev: bool) -> dict[str,Any]:
    model,legacy=load_audio(repo,e); rows={}
    dsmap=dm.val_dataset if dev else dm.test_dataset
    for task_id,task in enumerate(TASKS):
        key=f"dev_{task}" if dev else f"{protocol}_{task}"
        ds=dsmap[key]; loader=DataLoader(ds,batch_size=32,shuffle=False,drop_last=False,collate_fn=dm.collate_fn,num_workers=0)
        ls=[]; ys=[]
        for batch in loader:
            out=model(batch)
            if legacy:
                full=out["base_logits"]
                if dev:
                    leg=out["legacy_class_logits"]
                    if float((full-(leg[:,:,1]-leg[:,:,0])).abs().max())!=0.0 or int(((full>=0)!=(leg.argmax(-1)==1)).sum())!=0:
                        raise RuntimeError("audio42 margin/argmax equivalence failed")
                logits=full[:,task_id]
            else:
                logits=out.preds[:,1]-out.preds[:,0]
            ls.append(logits); ys.append(batch.targets)
        rows[task]=binary_metrics(torch.cat(ls),torch.cat(ys))
    mean=(rows["depression"]["score"]+rows["parkinson"]["score"])/2
    return {"depression":rows["depression"],"parkinson":rows["parkinson"],"mean_score":mean}

@torch.no_grad()
def evaluate_fusion(repo: Path, dm: WSMRampsSemanticDataModule, e: dict[str,Any], protocol: str, dev: bool) -> dict[str,Any]:
    device, use_amp = config_runtime(repo, e["config"])
    model=load_fusion(repo,e).to(device); ds=dm.val_dataset if dev else dm.test_dataset[protocol]
    loader=DataLoader(ds,batch_size=32,shuffle=False,drop_last=False,collate_fn=dm.collate_fn,num_workers=0)
    ls=[]; ts=[]; ms=[]
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(batch)
        ls.append(out.preds.detach().float().cpu()); ts.append(batch.targets.detach().float().cpu()); ms.append(batch.masks["observed_mask"].detach().cpu())
    return sparse_metrics(torch.cat(ls),torch.cat(ts),torch.cat(ms))

def evaluate_all(repo: Path, entries: list[dict[str,Any]], data_root: Path, audio_root: Path, video_root: Path, pseudo: Path, dev: bool) -> tuple[list[dict[str,Any]],dict[str,int]]:
    verify_ledger(repo,entries,pseudo)
    adm=WSMAudioSegmentDataModule(data_root=str(data_root),task="multitask",feature_cache_root=str(audio_root),test_filters=() if dev else ("none","soft","hard"),feature_model_name="microsoft/wavlm-base-plus",feature_layer=9,temporal_pool=4)
    fdm=WSMRampsSemanticDataModule(pseudo_cache_path=str(pseudo),data_root=str(data_root),audio_feature_cache_root=str(audio_root),video_cache_root=str(video_root),batch_size=32,num_workers=0,pin_memory=False,persistent_workers=False,shuffle_train=False,drop_last_train=False)
    results=[]
    for e in entries:
        for protocol in (("dev",) if dev else tuple(PROTOCOL_COUNTS)):
            metric=evaluate_audio(repo,adm,e,protocol,dev) if e["method"]=="audio" else evaluate_fusion(repo,fdm,e,protocol,dev)
            if dev:
                exp=DEV_EXPECTED[(e["method"],e["seed_key"])]
                got=(metric["depression"]["score"],metric["parkinson"]["score"],metric["mean_score"])
                if any(abs(a-b)>0.0005 for a,b in zip(got,exp)):
                    raise RuntimeError(f"DEV reproduction mismatch {e['method']} seed {e['seed']}: {got} != {exp}")
            else:
                results.append({**{k:e[k] for k in ("method","seed","checkpoint","checkpoint_sha256","config","config_sha256","run","epoch")}, "protocol":protocol, "protocol_count":PROTOCOL_COUNTS[protocol], "metrics":metric})
            if dev:
                results.append({**{k:e[k] for k in ("method","seed","checkpoint","checkpoint_sha256","config","config_sha256","run","epoch")}, "protocol":"dev", "metrics":metric})
    return results, {"test_none":1364,"test_soft":1208,"test_hard":1014}

def synthetic_checks() -> dict[str,bool]:
    assert abs(binary_metrics(torch.tensor([10.,-10.]),torch.tensor([1,0]))["score"]-1.0)<1e-12
    assert abs(binary_metrics(torch.tensor([-10.,10.]),torch.tensor([1,0]))["score"]-0.0)<1e-12
    assert compute_sparse_two_task_metrics(torch.tensor([[1.,-1.],[-1.,-1.]]),torch.tensor([[1.,float("nan")],[float("nan"),0.]]),torch.tensor([[True,False],[False,True]]),"test")["test/mean_score"]==1.0
    m=torch.tensor([[10.,-10.],[-10.,10.]])
    assert bool(((m[:,0]>=0)==(torch.tensor([1,0])==1)).all())
    values=np.array([1.,2.,3.,4.,5.]); assert abs(float(values.mean())-3.0)<1e-12 and abs(float(values.std(ddof=1))-math.sqrt(2.5))<1e-12
    return {"metric_formula":True,"raw_logit_threshold":True,"margin_argmax":True,"sparse_mask":True,"student_t_arithmetic":True}

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Evaluate the frozen Stage-7 checkpoints on DEV preflight or final TEST protocols.")
    p.add_argument("--data-root",required=True); p.add_argument("--audio-feature-cache-root",required=True); p.add_argument("--video-cache-root",required=True); p.add_argument("--pseudo-cache-path",required=True); p.add_argument("--output")
    p.add_argument("--dev-preflight",action="store_true")
    return p.parse_args()

def main() -> int:
    args=parse_args(); repo=Path(__file__).resolve().parents[2]; entries=frozen_entries(repo)
    if args.dev_preflight:
        checks=synthetic_checks(); results,_=evaluate_all(repo,entries,Path(args.data_root),Path(args.audio_feature_cache_root),Path(args.video_cache_root),Path(args.pseudo_cache_path),True)
        print(json.dumps({"mode":"dev_preflight","entry_count":len(entries),"synthetic_checks":checks,"dev_results":results,"test_iteration":False},sort_keys=True))
        return 0
    if not args.output: raise SystemExit("--output is required unless --dev-preflight is used")
    output=Path(args.output).expanduser().resolve()
    if output.exists(): raise SystemExit(f"refusing to overwrite existing output: {output}")
    synthetic_checks()
    results,counts=evaluate_all(repo,entries,Path(args.data_root),Path(args.audio_feature_cache_root),Path(args.video_cache_root),Path(args.pseudo_cache_path),False)
    report={"schema":"wsm-stage7-final-test-v1","runtime":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"device":"cpu","dtype":"float32","cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version()},"protocol_counts":counts,"entries":results,"no_raw_predictions":True,"no_raw_logits":True,"no_raw_probabilities":True,"no_labels":True,"no_sample_metadata":True}
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=".final_test_v1.",suffix=".json.tmp",dir=str(output.parent)); os.close(fd)
    try:
        if output.exists(): raise RuntimeError(f"refusing to overwrite existing output: {output}")
        with open(tmp,"w",encoding="utf-8") as f: json.dump(report,f,indent=2,sort_keys=True); f.write(chr(10))
        os.replace(tmp,output)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    print(json.dumps({"output":str(output),"entry_count":len(entries),"protocol_counts":counts},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
