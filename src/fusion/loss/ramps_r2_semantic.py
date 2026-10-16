"""Deterministic fixed-CLIP semantic reliability utilities for RAMPS R2."""
from __future__ import annotations
import hashlib
from typing import Any
import torch
from .ramps_r1_teacher import binary_brier,binary_ece,binary_nll,fit_binary_temperature

def normalize_embeddings(x: torch.Tensor)->torch.Tensor:
 return torch.nn.functional.normalize(torch.as_tensor(x,dtype=torch.float64),dim=-1)

def masked_pool_clip_frames(video: torch.Tensor, mask: torch.Tensor)->torch.Tensor:
 x=normalize_embeddings(video); m=torch.as_tensor(mask,dtype=torch.bool).unsqueeze(-1); return normalize_embeddings((x*m).sum(1)/m.sum(1).clamp_min(1))

def normalize_prompt_bank(x: torch.Tensor)->torch.Tensor:
    if isinstance(x, torch.Tensor):
        embeddings = x
    else:
        embeddings = getattr(x, "text_embeds", None)
        if not isinstance(embeddings, torch.Tensor):
            embeddings = getattr(x, "pooler_output", None)
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("CLIP text output has no tensor embedding field")
    normalized=normalize_embeddings(embeddings)
    mean=normalized.mean(0)
    return mean/mean.norm().clamp_min(1e-12)

def semantic_cosine_margin(image, depression, neutral):
 image=normalize_embeddings(image); depression=normalize_embeddings(depression).reshape(-1); neutral=normalize_embeddings(neutral).reshape(-1); return image@depression-image@neutral

def fit_monotonic_semantic_calibrator(margin: torch.Tensor, targets: torch.Tensor)->dict[str,Any]:
 x=torch.as_tensor(margin,dtype=torch.float64).reshape(-1); y=torch.as_tensor(targets,dtype=torch.float64).reshape(-1)
 if x.numel()==0 or x.shape!=y.shape or not bool(torch.isfinite(x).all()&torch.isfinite(y).all()): raise ValueError("invalid semantic calibration data")
 log_scale=torch.zeros((),dtype=torch.float64,requires_grad=True); bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
 opt=torch.optim.LBFGS([log_scale,bias],lr=.5,max_iter=100,tolerance_grad=1e-10,tolerance_change=1e-12,line_search_fn="strong_wolfe")
 def closure():
  opt.zero_grad(); loss=torch.nn.functional.binary_cross_entropy_with_logits(torch.exp(log_scale)*x+bias,y); loss.backward(); return loss
 opt.step(closure); raw_scale=float(torch.exp(log_scale.detach())); raw_bias=float(bias.detach()); scale=min(max(raw_scale,.01),100.); deployed_bias=min(max(raw_bias,-20.),20.)
 return {"unconstrained_scale":raw_scale,"unconstrained_bias":raw_bias,"scale":scale,"bias":deployed_bias,"clamped":bool(scale!=raw_scale or deployed_bias!=raw_bias)}

def semantic_metrics(margin,targets,cal):
 logits=cal["scale"]*torch.as_tensor(margin,dtype=torch.float64)+cal["bias"]; return {"nll":binary_nll(logits,targets),"brier":binary_brier(logits,targets),"ece":binary_ece(logits,targets,15)}

def semantic_confidence(prob,positive):
 p=torch.as_tensor(prob,dtype=torch.float64); return p if positive else 1-p

def semantic_entropy(prob):
 p=torch.as_tensor(prob,dtype=torch.float64).clamp(1e-12,1-1e-12); return -(p*p.log()+(1-p)*(1-p).log())/torch.log(torch.tensor(2.,dtype=p.dtype))

def semantic_hash(x: torch.Tensor)->str: return hashlib.sha256(torch.as_tensor(x).detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def search_semantic_rules(records:dict[str,torch.Tensor],precision_target=.90,min_support=10):
 a,s,v=records["audio_prob"],records["semantic_prob"],records.get("video_prob"); y=records["labels"].bool(); ae,se,ve=records["audio_entropy"],records["semantic_entropy"],records.get("video_entropy")
 aud=[]; chosen={}
 for side_name,pos in (("positive",True),("negative",False)):
  class_n=int((y if pos else ~y).sum()); candidates=[]
  for family in ("S1","S2","S3"):
   for i in range(50):
    tc=round(.5+.01*i,2)
    entropies=(None,) if family=="S1" else (.25,.5,.75)
    oods=(None,) if family!="S2" else (.8,.9,.95)
    for te in entropies:
     for oo in oods:
      agreement=(a>=.5)==(s>=.5); side=(a>=.5) if pos else (a<.5); conf=torch.minimum(semantic_confidence(a,pos),semantic_confidence(s,pos)); mask=agreement&side&(conf>=tc)
      if family=="S2": mask &= torch.maximum(ae,se)<=te; mask &= (records["ood_percentile_positive"] if pos else records["ood_percentile_negative"])<=oo
      if family=="S3":
       agreement3=agreement & ((v>=.5)==(s>=.5)); conf=torch.minimum(conf,semantic_confidence(v,pos)); mask=agreement3&side&(conf>=tc)&(torch.maximum(torch.maximum(ae,se),ve)<=te)
      support=int(mask.sum()); correct=int((mask&(y if pos else ~y)).sum()); precision=correct/support if support else 0.; row={"side":side_name,"family":family,"tau_conf":tc,"tau_entropy":te,"tau_ood":oo,"support":support,"correct":correct,"precision":precision,"coverage":correct/class_n if class_n else 0.,"eligible":support>=min_support and precision>=precision_target}; aud.append(row); candidates.append(row)
  eligible=[r for r in candidates if r["eligible"]]
  chosen[side_name]=(sorted(eligible,key=lambda r:(-r["support"],-r["precision"],{"S1":0,"S2":1,"S3":2}[r["family"]],r["tau_conf"],-(r["tau_entropy"] if r["tau_entropy"] is not None else 1.),-(r["tau_ood"] if r["tau_ood"] is not None else 1.)))[0] if eligible else {"enabled":False,"side":side_name})
  if chosen[side_name].get("enabled",True): chosen[side_name]["enabled"]=True
 return chosen,aud

def apply_semantic_rule(records,rule,pos):
 if not rule.get("enabled",False): return torch.zeros_like(records["audio_prob"],dtype=torch.bool)
 a,s,v=records["audio_prob"],records["semantic_prob"],records.get("video_prob"); agreement=(a>=.5)==(s>=.5); side=(a>=.5) if pos else (a<.5); conf=torch.minimum(semantic_confidence(a,pos),semantic_confidence(s,pos)); mask=agreement&side&(conf>=rule["tau_conf"])
 if rule["family"]=="S2": mask &= torch.maximum(records["audio_entropy"],records["semantic_entropy"])<=rule["tau_entropy"]; mask &= (records["ood_percentile_positive"] if pos else records["ood_percentile_negative"])<=rule["tau_ood"]
 if rule["family"]=="S3": mask &= ((v>=.5)==(s>=.5)); mask &= torch.minimum(conf,semantic_confidence(v,pos))>=rule["tau_conf"]; mask &= torch.maximum(torch.maximum(records["audio_entropy"],records["semantic_entropy"]),records["video_entropy"])<=rule["tau_entropy"]
 return mask

def semantic_reliability(records, rule, positive, mask):
    """Detached candidate-side semantic reliability for one depression class side."""
    audio_conf=semantic_confidence(records["audio_prob"],positive)
    semantic_conf=semantic_confidence(records["semantic_prob"],positive)
    value=torch.minimum(audio_conf,semantic_conf)
    if rule.get("family")=="S2":
        ood=records["ood_percentile_positive"] if positive else records["ood_percentile_negative"]
        value=value*(1-torch.maximum(records["audio_entropy"],records["semantic_entropy"]))*(1-ood)
    if rule.get("family")=="S3":
        video_conf=semantic_confidence(records["video_prob"],positive)
        value=torch.minimum(value,video_conf)*(1-torch.maximum(torch.maximum(records["audio_entropy"],records["semantic_entropy"]),records["video_entropy"]))
    return value.detach().clamp(0,1)*torch.as_tensor(mask,dtype=torch.bool)
