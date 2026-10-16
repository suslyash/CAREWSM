"""Offline RAMPS-R2 reliability search with frozen audio and independent video teachers."""
from __future__ import annotations
import argparse, hashlib, json, random, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader
from fusion.data.wsm_av_fusion_datamodule import WSMAVFusionDataModule
from fusion.loss.ramps_r1_teacher import fit_binary_temperature
from fusion.loss.ramps_r2_reliability import (audio_centroid_distance, detached_reliability,
    empirical_ood_percentile, normalized_audio_class_centroids, normalized_binary_entropy,
    select_reliability_rules, stratified_folds, apply_reliability_rule)
from fusion.models.frozen_audio_temporal_adapter import FrozenAudioTemporalAdapter
from video.models.depart_v2 import WSMVideoDepartV2Model
from common.callbacks.wsm_segment_callback import compute_sparse_two_task_metrics

TASKS=("depression","parkinson")
AUDIO_SHA="0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2"
EXPECTED={"train":6325,"dev":933,"test_none":1364,"test_soft":1208,"test_hard":1014}

def sha256(path: Path)->str:
 h=hashlib.sha256();
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
 return h.hexdigest()

def atomic_json(path: Path, value: Any)->None:
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,prefix=".tmp-",suffix=".json",delete=False) as f:
  json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); name=f.name
 Path(name).replace(path)

def infer(teacher_a, teacher_v, dataset, collate, device, batch_size, workers):
 loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,collate_fn=collate,pin_memory=True)
 ids=[]; targets=[]; observed=[]; al=[]; af=[]; vl=[]
 with torch.inference_mode():
  for batch in loader:
   meta=batch.meta["sample_meta"]
   ids += [str(x["segment_id"]) for x in meta]
   targets.append(batch.targets.cpu()); observed.append(batch.get_masks("observed_mask").cpu())
   batch.inputs={key:value.to(device) if isinstance(value,torch.Tensor) else value for key,value in batch.inputs.items()}
   batch.masks={key:value.to(device) if isinstance(value,torch.Tensor) else value for key,value in batch.masks.items()}
   b=batch
   ao=teacher_a(b); vo=teacher_v(b)
   al.append(ao["base_logits"].detach().cpu()); af.append(ao["task_features"].detach().cpu()); vl.append(vo.preds.detach().cpu())
 return {"segment_ids":ids,"targets":torch.cat(targets),"observed":torch.cat(observed),"audio_logits":torch.cat(al),"audio_features":torch.cat(af),"video_logits":torch.cat(vl)}

def deployment(records, task, indices):
 y=records["targets"][indices,task]; a=records["audio_logits"][indices,task]; v=records["video_logits"][indices,task]; f=records["audio_features"][indices,task]
 ta=fit_binary_temperature(a[y.isfinite()],y[y.isfinite()])["temperature"]
 tv=fit_binary_temperature(v[y.isfinite()],y[y.isfinite()])["temperature"]
 labels=y.to(torch.long); cent=normalized_audio_class_centroids(f,labels)
 distances=audio_centroid_distance(f,cent,labels)
 refs={c:distances[labels==c] for c in (0,1)}
 return float(ta),float(tv),cent,refs

def oof_for_task(records, task, name, folds, precision_target, min_support):
 obs=records["observed"][:,task]; idx=torch.where(obs)[0]; labels=records["targets"][idx,task].to(torch.long)
 ids=[records["segment_ids"][int(i)] for i in idx]
 fold=stratified_folds(name,ids,labels,5)
 oof={"audio_prob":torch.zeros(len(idx),dtype=torch.float64),"video_prob":torch.zeros(len(idx),dtype=torch.float64),"audio_entropy":torch.zeros(len(idx),dtype=torch.float64),"video_entropy":torch.zeros(len(idx),dtype=torch.float64),"ood_percentile":torch.zeros(len(idx),dtype=torch.float64),"labels":labels}
 fold_audit=[]
 for k in range(5):
  hold=fold==k; fit=~hold
  if not bool((labels[fit]==0).any() and (labels[fit]==1).any()): raise RuntimeError(f"R2 blocked: {name} fit fold {k} lacks a class")
  fit_i=idx[fit]; hold_i=idx[hold]
  ta,tv,cent,refs=deployment(records,task,fit_i)
  pa=torch.sigmoid(records["audio_logits"][hold_i,task].double()/ta); pv=torch.sigmoid(records["video_logits"][hold_i,task].double()/tv)
  feats=records["audio_features"][hold_i,task].double(); labs=records["targets"][hold_i,task].long()
  d=audio_centroid_distance(feats,cent,labs); pct=torch.zeros_like(d)
  for c in (0,1):
   q=labs==c
   if bool(q.any()): pct[q]=empirical_ood_percentile(d[q],refs[c])
  pos=hold.nonzero().flatten(); oof["audio_prob"][pos]=pa; oof["video_prob"][pos]=pv; oof["audio_entropy"][pos]=normalized_binary_entropy(pa); oof["video_entropy"][pos]=normalized_binary_entropy(pv); oof["ood_percentile"][pos]=pct
  fold_audit.append({"fold":k,"fit_rows":int(fit.sum()),"heldout_rows":int(hold.sum()),"fit_class_0":int((labels[fit]==0).sum()),"fit_class_1":int((labels[fit]==1).sum()),"heldout_class_0":int((labels[hold]==0).sum()),"heldout_class_1":int((labels[hold]==1).sum()),"audio_temperature":ta,"video_temperature":tv})
 rules,audits=select_reliability_rules(oof,precision_target=precision_target,min_support=min_support)
 return oof,fold_audit,rules,audits,idx

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--data-root",required=True); ap.add_argument("--audio-feature-cache-root",required=True); ap.add_argument("--video-cache-root",required=True); ap.add_argument("--audio-checkpoint",required=True); ap.add_argument("--video-checkpoint",required=True); ap.add_argument("--output-root",required=True); ap.add_argument("--precision-target",type=float,default=.90); ap.add_argument("--min-support",type=int,default=10); ap.add_argument("--folds",type=int,default=5); ap.add_argument("--batch-size",type=int,default=32); ap.add_argument("--num-workers",type=int,default=4); ap.add_argument("--device",default="cuda"); ap.add_argument("--overwrite",action="store_true"); args=ap.parse_args()
 if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable; R2 production is blocked")
 if args.folds!=5 or args.precision_target<.90 or args.min_support<10: raise ValueError("R2 fixed fold/precision/support contract was changed")
 out=Path(args.output_root).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()) and not args.overwrite: raise FileExistsError(f"non-empty output root refuses overwrite: {out}")
 ac=Path(args.audio_checkpoint); vc=Path(args.video_checkpoint); audio_hash=sha256(ac); video_hash=sha256(vc)
 if audio_hash!=AUDIO_SHA: raise RuntimeError("audio checkpoint SHA mismatch")
 payload=torch.load(vc,map_location="cpu",weights_only=False)
 if not isinstance(payload,dict) or payload.get("epoch")!=5: raise RuntimeError("video checkpoint payload is not epoch 5")
 model=WSMVideoDepartV2Model(video_feature_dim=512,hidden_dim=192,num_layers=2,num_heads=4,ff_mult=4,dropout=.2,sequence_steps=60,num_tasks=2,prototype_scale=10.,gate_hidden_dim=64)
 model.load_state_dict(payload["model_state_dict"],strict=True); model.eval()
 for p in model.parameters(): p.requires_grad_(False)
 audio=FrozenAudioTemporalAdapter(str(ac)); audio.eval()
 if any(p.requires_grad for p in list(audio.parameters())+list(model.parameters())): raise RuntimeError("teacher is not frozen")
 data=WSMAVFusionDataModule(data_root=args.data_root,audio_feature_cache_root=args.audio_feature_cache_root,video_cache_root=args.video_cache_root,batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=True,persistent_workers=False,shuffle_train=False)
 if data.audit["counts"]!=EXPECTED: raise RuntimeError(f"canonical counts failed: {data.audit}")
 device=torch.device(args.device); audio.to(device); model.to(device)
 dev=infer(audio,model,data.val_dataset,data.collate_fn,device,args.batch_size,args.num_workers)
 metrics_a=compute_sparse_two_task_metrics(dev["audio_logits"],dev["targets"],dev["observed"],"dev",TASKS); metrics_v=compute_sparse_two_task_metrics(dev["video_logits"],dev["targets"],dev["observed"],"dev",TASKS)
 expected_a={"depression":.7479183895,"parkinson":.8277353635,"mean_score":.7878268765}; expected_v={"depression":.620101,"parkinson":.793043,"mean_score":.706572}
 got_a={"depression":metrics_a["dev/depression/score"],"parkinson":metrics_a["dev/parkinson/score"],"mean_score":metrics_a["dev/mean_score"]}; got_v={"depression":metrics_v["dev/depression/score"],"parkinson":metrics_v["dev/parkinson/score"],"mean_score":metrics_v["dev/mean_score"]}
 if any(abs(got_v[k]-expected_v[k])>.0005 for k in expected_v): raise RuntimeError(f"video DEV reproduction failed: {got_v}")
 if any(abs(got_a[k]-expected_a[k])>.0005 for k in expected_a): raise RuntimeError(f"audio DEV reproduction failed: {got_a}")
 search={"version":"ramps-r2-av-reliability-v1","generated_utc":datetime.now(timezone.utc).isoformat(),"audio_checkpoint_path":str(ac.resolve()),"audio_checkpoint_sha256":audio_hash,"video_checkpoint_path":str(vc.resolve()),"video_checkpoint_sha256":video_hash,"audio_teacher":"FrozenAudioTemporalAdapter","video_teacher":"WSMVideoDepartV2Model","data_root":str(Path(args.data_root).resolve()),"audio_feature_cache_root":str(Path(args.audio_feature_cache_root).resolve()),"video_cache_root":str(Path(args.video_cache_root).resolve()),"counts":data.audit["counts"],"no_test_statement":"Only train_dataset and val_dataset were iterated; no Test rows or metrics were used.","historical_dev_reproduction":{"audio":got_a,"video":got_v},"precision_target":args.precision_target,"min_support":args.min_support,"folds":5,"families":{"A":"agreement + joint confidence","B":"A + uncertainty <= tau_entropy","C":"B + audio OOD percentile <= tau_ood"},"tasks":{}}
 all_rules={}; deployments={}
 for task,name in enumerate(TASKS):
  oof,fold_audit,rules,audits,idx=oof_for_task(dev,task,name,5,args.precision_target,args.min_support); search["tasks"][name]={"folds":fold_audit,"oof_candidate_audit":audits,"selected_oof_rules":rules}; all_rules[name]=rules
  ta,tv,cent,refs=deployment(dev,task,torch.where(dev["observed"][:,task])[0]); deployments[name]={"audio_temperature":ta,"video_temperature":tv,"centroids":cent,"refs":refs}
  confirmed={}
  for side,positive in (("positive",True),("negative",False)):
   rule=rules[side]
   if not rule.get("enabled",False): confirmed[side]={"enabled":False}; continue
   ids=torch.where(dev["observed"][:,task])[0]; rec={"audio_prob":torch.sigmoid(dev["audio_logits"][ids,task].double()/ta),"video_prob":torch.sigmoid(dev["video_logits"][ids,task].double()/tv),"audio_entropy":normalized_binary_entropy(torch.sigmoid(dev["audio_logits"][ids,task].double()/ta)),"video_entropy":normalized_binary_entropy(torch.sigmoid(dev["video_logits"][ids,task].double()/tv)),"ood_percentile":torch.zeros(len(ids),dtype=torch.float64),"labels":dev["targets"][ids,task].long()}; d=audio_centroid_distance(dev["audio_features"][ids,task].double(),cent,rec["labels"])
   for c in (0,1):
    q=rec["labels"]==c
    if bool(q.any()): rec["ood_percentile"][q]=empirical_ood_percentile(d[q],refs[c])
   mask=apply_reliability_rule(rec,rule,positive); lab=rec["labels"]; correct=int((mask&(lab==int(positive))).sum()); support=int(mask.sum()); confirmed[side]={**rule,"support_full_dev":support,"correct_full_dev":correct,"precision_full_dev":correct/support if support else 0.,"coverage_full_dev":correct/int((lab==int(positive)).sum()),"enabled":support>=args.min_support and correct/support>=args.precision_target if support else False}
  search["tasks"][name].update({"full_dev_audio_temperature":ta,"full_dev_video_temperature":tv,"full_dev_confirmation":confirmed,"deployable":any(x.get("enabled",False) for x in confirmed.values())}); deployments[name]["rules"]=confirmed
 search["overall_two_head_gate"]=all(x["deployable"] for x in search["tasks"].values())
 atomic_json(out/"reliability_search.json",search)
 if not search["overall_two_head_gate"]:
  task_audit={}
  for task,name in enumerate(TASKS):
   missing=int((~data.train_dataset._samples[0]["observed_mask"]).sum()) if False else sum(int((~sample["observed_mask"][task]).item()) for sample in data.train_dataset._samples)
   task_audit[name]={"missing_train_count":missing,"accepted_total":0,"rejected_total":missing,"coverage":0.0,"accepted_positive_count":0,"accepted_negative_count":0,"selected_rules":deployments[name]["rules"],"agreement_rate_missing":None,"accepted_probability_stats":None,"reliability_stats":None,"deployable":search["tasks"][name]["deployable"]}
  audit={"version":"ramps-r2-av-reliability-v1","blocked":True,"reason":"overall two-head DEV reliability gate failed","train_inference_ran":False,"no_missing_label_correctness_claim":True,"tasks":task_audit,"counts":data.audit["counts"],"train_rows":6325,"observed_overwrite_violations":0,"duplicate_segment_ids":0,"nonfinite_accepted_targets":0,"pseudo_values_on_observed_entries":0,"pseudo_acceptance_on_observed_entries":0,"no_test_statement":"No Test rows or metrics were iterated or inspected."}
  atomic_json(out/"audit.json",audit); raise RuntimeError("R2 blocked: one or more diseases have no deployable reliability side")
 train=infer(audio,model,data.train_dataset,data.collate_fn,device,args.batch_size,args.num_workers)
 pseudo_accept=torch.zeros((6325,2),dtype=torch.bool); pseudo_targets=torch.full((6325,2),float("nan")); pseudo_rel=torch.zeros((6325,2)); pseudo_class=torch.full((6325,2),-1,dtype=torch.long)
 for task,name in enumerate(TASKS):
  dep=deployments[name]; probs_a=torch.sigmoid(train["audio_logits"][:,task].double()/dep["audio_temperature"]); probs_v=torch.sigmoid(train["video_logits"][:,task].double()/dep["video_temperature"]); labels=train["targets"][:,task].long(); features=train["audio_features"][:,task].double(); distances=torch.zeros(6325,dtype=torch.float64)
  for c in (0,1):
   q=labels==c
   if bool(q.any()): distances[q]=audio_centroid_distance(features[q],dep["centroids"],labels[q])
  rec={"audio_prob":probs_a,"video_prob":probs_v,"audio_entropy":normalized_binary_entropy(probs_a),"video_entropy":normalized_binary_entropy(probs_v),"ood_percentile":torch.zeros(6325,dtype=torch.float64),"labels":labels}
  for c in (0,1):
   q=labels==c
   if bool(q.any()): rec["ood_percentile"][q]=empirical_ood_percentile(distances[q],dep["refs"][c])
  for side,positive in (("positive",True),("negative",False)):
   rule=dep["rules"][side]; mask=apply_reliability_rule(rec,rule,positive)&~train["observed"][:,task]; pseudo_accept[:,task]|=mask; pseudo_targets[mask,task]=probs_a[mask]; pseudo_rel[mask,task]=detached_reliability(rec,rule,mask)[mask]; pseudo_class[mask,task]=int(positive)
 artifact={"version":"ramps-r2-av-reliability-v1","task_names":list(TASKS),"segment_ids":train["segment_ids"],"observed_mask":train["observed"],"observed_targets":train["targets"],"raw_audio_logits":train["audio_logits"],"calibrated_audio_probs":torch.stack([torch.sigmoid(train["audio_logits"][:,i].double()/deployments[TASKS[i]]["audio_temperature"]) for i in range(2)],1).float(),"raw_video_logits":train["video_logits"],"calibrated_video_probs":torch.stack([torch.sigmoid(train["video_logits"][:,i].double()/deployments[TASKS[i]]["video_temperature"]) for i in range(2)],1).float(),"audio_task_features":train["audio_features"],"pseudo_accept_mask":pseudo_accept,"pseudo_targets":pseudo_targets,"pseudo_reliability":pseudo_rel,"pseudo_class":pseudo_class,"selected_rules":{n:deployments[n]["rules"] for n in TASKS},"audio_temperatures":{n:deployments[n]["audio_temperature"] for n in TASKS},"video_temperatures":{n:deployments[n]["video_temperature"] for n in TASKS},"audio_checkpoint_path":str(ac.resolve()),"audio_checkpoint_sha256":audio_hash,"video_checkpoint_path":str(vc.resolve()),"video_checkpoint_sha256":video_hash}
 torch.save(artifact,out/"train_missing_targets.pt")
 audit={"version":"ramps-r2-av-reliability-v1","blocked":False,"train_inference_ran":True,"no_missing_label_correctness_claim":True,"counts":data.audit["counts"],"train_rows":6325,"observed_overwrite_violations":int((pseudo_accept&train["observed"]).sum()),"duplicate_segment_ids":len(train["segment_ids"])-len(set(train["segment_ids"])),"nonfinite_accepted_targets":int((~torch.isfinite(pseudo_targets[pseudo_accept])).sum()),"pseudo_values_on_observed_entries":int(torch.isfinite(pseudo_targets[train["observed"]]).sum()),"pseudo_acceptance_on_observed_entries":int((pseudo_accept&train["observed"]).sum()),"no_test_statement":"Only train_dataset and val_dataset were iterated; no Test rows or metrics were used.","tasks":{}}
 for task,name in enumerate(TASKS):
  missing=~train["observed"][:,task]; accepted=pseudo_accept[:,task]; vals=pseudo_targets[accepted,task]; rel=pseudo_rel[accepted,task]; audit["tasks"][name]={"missing_train_count":int(missing.sum()),"accepted_total":int(accepted.sum()),"rejected_total":int((missing&~accepted).sum()),"coverage":float(accepted.sum()/missing.sum()) if bool(missing.any()) else 0.0,"accepted_positive_count":int((accepted&(pseudo_class[:,task]==1)).sum()),"accepted_negative_count":int((accepted&(pseudo_class[:,task]==0)).sum()),"selected_rules":deployments[name]["rules"],"accepted_probability_stats":{"min":float(vals.min()) if vals.numel() else None,"max":float(vals.max()) if vals.numel() else None,"mean":float(vals.mean()) if vals.numel() else None},"reliability_stats":{"min":float(rel.min()) if rel.numel() else None,"max":float(rel.max()) if rel.numel() else None,"mean":float(rel.mean()) if rel.numel() else None}}
 atomic_json(out/"audit.json",audit)
 print(json.dumps({"status":"passed","output_root":str(out),"accepted_missing":int(pseudo_accept.sum())},sort_keys=True))

if __name__=="__main__": main()
