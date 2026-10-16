"""Deterministic offline RAMPS-R2 reliability utilities."""
from __future__ import annotations
import hashlib
from typing import Any
import torch


def normalized_binary_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    p=torch.as_tensor(probabilities,dtype=torch.float64).clamp(1e-12,1-1e-12)
    return -(p*p.log()+(1-p)*(1-p).log())/torch.log(torch.tensor(2.0,dtype=p.dtype))


def stratified_folds(task_name: str, segment_ids: list[str], labels: torch.Tensor, folds: int=5) -> torch.Tensor:
    labels=torch.as_tensor(labels,dtype=torch.long).reshape(-1)
    if len(segment_ids)!=labels.numel() or folds<2 or not bool(torch.logical_or(labels==0,labels==1).all()): raise ValueError("invalid stratified-fold inputs")
    out=torch.full((labels.numel(),),-1,dtype=torch.long)
    for cls in (0,1):
        indices=[i for i,v in enumerate(labels.tolist()) if v==cls]
        indices.sort(key=lambda i: hashlib.sha256(f"{task_name}:{segment_ids[i]}".encode()).hexdigest())
        for rank,i in enumerate(indices): out[i]=rank%folds
    if bool((out<0).any()) or not bool(torch.bincount(out,minlength=folds).gt(0).all()): raise ValueError("fold assignment failed")
    return out


def class_confidence(probabilities: torch.Tensor, positive: bool) -> torch.Tensor:
    p=torch.as_tensor(probabilities,dtype=torch.float64)
    return p if positive else 1-p


def same_class_agreement(audio_prob: torch.Tensor, video_prob: torch.Tensor, positive: bool) -> torch.Tensor:
    a=torch.as_tensor(audio_prob)>=0.5; v=torch.as_tensor(video_prob)>=0.5
    side=a if positive else ~a
    return (a==v)&side


def normalized_audio_class_centroids(features: torch.Tensor, labels: torch.Tensor) -> dict[int,torch.Tensor]:
    features=torch.as_tensor(features,dtype=torch.float64); labels=torch.as_tensor(labels,dtype=torch.long)
    result={}
    for cls in (0,1):
        x=features[labels==cls]
        if x.shape[0]==0: raise ValueError("class centroid has no support")
        result[cls]=torch.nn.functional.normalize(x.mean(0),dim=0)
    return result


def audio_centroid_distance(features: torch.Tensor, centroids: dict[int,torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    x=torch.nn.functional.normalize(torch.as_tensor(features,dtype=torch.float64),dim=1)
    c=torch.stack([centroids[int(v)] for v in torch.as_tensor(labels).tolist()])
    return 1-(x*c).sum(1)


def empirical_ood_percentile(distances: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    d=torch.as_tensor(distances,dtype=torch.float64).reshape(-1); r=torch.as_tensor(reference,dtype=torch.float64).reshape(-1)
    if not r.numel(): raise ValueError("OOD reference is empty")
    return (r[None,:] <= d[:,None]).to(torch.float64).mean(1)


def select_reliability_rules(records: dict[str,torch.Tensor], *, precision_target: float=0.90, min_support: int=10) -> tuple[dict[str,Any],list[dict[str,Any]]]:
    p_a,p_v=records["audio_prob"],records["video_prob"]; labels=records["labels"].to(torch.bool)
    agreement=(p_a>=0.5)==(p_v>=0.5); conf=torch.minimum(torch.where(p_a>=0.5,p_a,1-p_a),torch.where(p_v>=0.5,p_v,1-p_v))
    unc=torch.maximum(records["audio_entropy"],records["video_entropy"]); ood=records["ood_percentile"]
    audits=[]; chosen={}
    for side_name,positive in (("positive",True),("negative",False)):
        rows=[]; class_count=int((labels if positive else ~labels).sum())
        for family in ("A","B","C"):
            for tau_conf_i in range(50):
                tau_conf=0.50+0.01*tau_conf_i
                for tau_entropy in ((None,) if family=="A" else (0.25,0.50,0.75)):
                    for tau_ood in ((None,) if family!="C" else (0.80,0.90,0.95)):
                        side=(p_a>=0.5) if positive else (p_a<0.5)
                        mask=agreement&side&(conf>=tau_conf)
                        if tau_entropy is not None: mask &= unc<=tau_entropy
                        if tau_ood is not None: mask &= ood<=tau_ood
                        support=int(mask.sum()); correct=int((mask & (labels if positive else ~labels)).sum())
                        precision=correct/support if support else 0.0
                        row={"side":side_name,"family":family,"tau_conf":round(tau_conf,2),"tau_entropy":tau_entropy,"tau_ood":tau_ood,"support":support,"correct":correct,"precision":precision,"coverage":correct/class_count if class_count else 0.0,"eligible":support>=min_support and precision>=precision_target}
                        rows.append(row); audits.append(row)
        eligible=[x for x in rows if x["eligible"]]
        if eligible:
            chosen[side_name]=sorted(eligible,key=lambda x:(-x["support"],-x["precision"],{"A":0,"B":1,"C":2}[x["family"]],x["tau_conf"],-(x["tau_entropy"] if x["tau_entropy"] is not None else 1.0),-(x["tau_ood"] if x["tau_ood"] is not None else 1.0)))[0]
        else: chosen[side_name]={"enabled":False,"side":side_name}
        if side_name in chosen and chosen[side_name].get("enabled",True): chosen[side_name]["enabled"]=True
    return chosen,audits


def apply_reliability_rule(records: dict[str,torch.Tensor], rule: dict[str,Any], positive: bool) -> torch.Tensor:
    if not rule.get("enabled",False): return torch.zeros_like(records["audio_prob"],dtype=torch.bool)
    a,v=records["audio_prob"],records["video_prob"]; agreement=(a>=0.5)==(v>=0.5); side=(a>=0.5) if positive else (a<0.5)
    mask=agreement&side&(torch.minimum(torch.where(a>=0.5,a,1-a),torch.where(v>=0.5,v,1-v))>=rule["tau_conf"])
    if rule["tau_entropy"] is not None: mask &= torch.maximum(records["audio_entropy"],records["video_entropy"])<=rule["tau_entropy"]
    if rule["tau_ood"] is not None: mask &= records["ood_percentile"]<=rule["tau_ood"]
    return mask


def detached_reliability(records: dict[str,torch.Tensor], rule: dict[str,Any], mask: torch.Tensor) -> torch.Tensor:
    a,v=records["audio_prob"],records["video_prob"]; joint=torch.minimum(torch.where(a>=0.5,a,1-a),torch.where(v>=0.5,v,1-v))
    value=joint
    if rule.get("family") in ("B","C"): value=value*(1-torch.maximum(records["audio_entropy"],records["video_entropy"]))
    if rule.get("family")=="C": value=value*(1-records["ood_percentile"])
    return value.detach().clamp(0,1)*mask
