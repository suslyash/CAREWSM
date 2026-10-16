"""Offline fixed-CLIP semantic evidence audit for RAMPS R2."""
from __future__ import annotations
import argparse,hashlib,json,tempfile,random
from datetime import datetime,timezone
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from fusion.data.wsm_av_fusion_datamodule import WSMAVFusionDataModule
from fusion.loss.ramps_r1_teacher import fit_binary_temperature
from fusion.loss.ramps_r2_reliability import normalized_audio_class_centroids,audio_centroid_distance,empirical_ood_percentile,stratified_folds
from fusion.loss.ramps_r2_semantic import *
from fusion.models.frozen_audio_temporal_adapter import FrozenAudioTemporalAdapter
from video.models.depart_v2 import WSMVideoDepartV2Model
from common.callbacks.wsm_segment_callback import compute_sparse_two_task_metrics
TASKS=("depression","parkinson"); AUDIO_SHA="0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2"; VIDEO_SHA="3e39778126db401a2fe17bb472a172191616e8a7b5265e982233c53dcd4af2f6"; EXPECTED={"train":6325,"dev":933,"test_none":1364,"test_soft":1208,"test_hard":1014}
PROMPTS={"depression":["a video of a person with depression","a person with depressive symptoms","a person showing signs of depression"],"neutral":["a video of a person","a person","a video showing a person"]}
def sha(p):
 h=hashlib.sha256(); f=open(p,'rb')
 for x in iter(lambda:f.read(1<<20),b''): h.update(x)
 f.close(); return h.hexdigest()
def atomic(path,obj):
 with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.tmp-',suffix='.json',delete=False,encoding='utf8') as f: json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False); n=f.name
 Path(n).replace(path)
def atomic_torch(path,obj):
 with tempfile.NamedTemporaryFile('wb',dir=path.parent,prefix='.tmp-',suffix='.pt',delete=False) as f:
  torch.save(obj,f); f.flush(); name=f.name
 Path(name).replace(path)

def stats(value):
 value=value.detach().cpu().flatten()
 return {'min':float(value.min()) if value.numel() else None,'max':float(value.max()) if value.numel() else None,'mean':float(value.mean()) if value.numel() else None}

def move_batch(b,device):
 b.inputs={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in b.inputs.items()}; b.masks={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in b.masks.items()}; return b
def infer(audio,video,clip,dep_text,neu_text,dataset,collate,device,bs,nw):
 loader=DataLoader(dataset,batch_size=bs,shuffle=False,num_workers=nw,collate_fn=collate,pin_memory=True); out={k:[] for k in ('targets','observed','audio_logits','audio_features','video_logits','margin')}; ids=[]
 with torch.inference_mode():
  for b in loader:
   ids += [str(x['segment_id']) for x in b.meta['sample_meta']]; out['targets'].append(b.targets.cpu()); out['observed'].append(b.get_masks('observed_mask').cpu()); b=move_batch(b,device); ao=audio(b); vo=video(b); image=masked_pool_clip_frames(b.inputs['video'],b.masks['video_mask']); out['audio_logits'].append(ao['base_logits'].cpu()); out['audio_features'].append(ao['task_features'].cpu()); out['video_logits'].append(vo.preds.cpu()); out['margin'].append(semantic_cosine_margin(image,dep_text,neu_text).cpu())
 out={k:torch.cat(v) for k,v in out.items()}; out['segment_ids']=ids; return out
def fit_stats(r,idx,task):
 y=r['targets'][idx,task]; good=y.isfinite(); ii=idx[good]; ta=fit_binary_temperature(r['audio_logits'][ii,task],y[good])['temperature']; tv=fit_binary_temperature(r['video_logits'][ii,task],y[good])['temperature'];
 cal=fit_monotonic_semantic_calibrator(r['margin'][ii],y[good]); lab=y[good].long(); cent=normalized_audio_class_centroids(r['audio_features'][ii,task].double(),lab); d=audio_centroid_distance(r['audio_features'][ii,task].double(),cent,lab); refs={c:d[lab==c] for c in (0,1)}; return ta,tv,cal,cent,refs
def records(r,idx,task,stats):
 ta,tv,cal,cent,refs=stats; a=torch.sigmoid(r['audio_logits'][idx,task].double()/ta); v=torch.sigmoid(r['video_logits'][idx,task].double()/tv); s=torch.sigmoid(cal['scale']*r['margin'][idx].double()+cal['bias']); lab=r['targets'][idx,task].long(); dists={}
 for c in (0,1):
  all_labels=torch.full((len(idx),),c,dtype=torch.long); dd=audio_centroid_distance(r['audio_features'][idx,task].double(),cent,all_labels); dists[c]=empirical_ood_percentile(dd,refs[c])
 return {'audio_prob':a,'video_prob':v,'semantic_prob':s,'audio_entropy':semantic_entropy(a),'video_entropy':semantic_entropy(v),'semantic_entropy':semantic_entropy(s),'ood_percentile_positive':dists[1],'ood_percentile_negative':dists[0],'labels':lab}
def main():
 ap=argparse.ArgumentParser();
 for n in ('data_root','audio_feature_cache_root','video_cache_root','audio_checkpoint','video_checkpoint','output_root'): ap.add_argument('--'+n.replace('_','-'),required=True)
 ap.add_argument('--clip-model',default='openai/clip-vit-base-patch32'); ap.add_argument('--clip-revision',default='main'); ap.add_argument('--precision-target',type=float,default=.9); ap.add_argument('--min-support',type=int,default=10); ap.add_argument('--folds',type=int,default=5); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--num-workers',type=int,default=4); ap.add_argument('--device',default='cuda'); ap.add_argument('--overwrite',action='store_true'); args=ap.parse_args()
 random.seed(42); torch.manual_seed(42)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
 if args.clip_model!='openai/clip-vit-base-patch32' or args.clip_revision!='main' or args.precision_target!=0.90 or args.min_support!=10 or args.folds!=5: raise ValueError('fixed TASK-005A4-C1 contract changed')
 if not torch.cuda.is_available() and args.device.startswith('cuda'): raise RuntimeError('CUDA unavailable; semantic audit blocked')
 out=Path(args.output_root).resolve(); out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()) and not args.overwrite: raise FileExistsError('non-empty output root refuses overwrite')
 ac,vc=Path(args.audio_checkpoint),Path(args.video_checkpoint); ah,vh=sha(ac),sha(vc)
 if ah!=AUDIO_SHA or vh!=VIDEO_SHA: raise RuntimeError('teacher checkpoint SHA mismatch')
 vp=torch.load(vc,map_location='cpu',weights_only=False)
 if vp.get('epoch')!=5: raise RuntimeError('video epoch mismatch')
 video=WSMVideoDepartV2Model(video_feature_dim=512,hidden_dim=192,num_layers=2,num_heads=4,ff_mult=4,dropout=.2,sequence_steps=60,num_tasks=2,prototype_scale=10.,gate_hidden_dim=64); video.load_state_dict(vp['model_state_dict'],strict=True); video.eval()
 audio=FrozenAudioTemporalAdapter(str(ac)); audio.eval()
 prompt_payload={'version':'ramps-r2-semantic-v1','clip_model':args.clip_model,'clip_revision':args.clip_revision,'prompts':PROMPTS,'no_sample_specific_information':True,'no_negative_prompts':True}; prompt_bytes=json.dumps(prompt_payload,sort_keys=True,separators=(',',':')).encode(); prompt_payload['canonical_json_sha256']=hashlib.sha256(prompt_bytes).hexdigest()
 try:
  from transformers import CLIPModel,CLIPProcessor
  processor=CLIPProcessor.from_pretrained(args.clip_model,revision=args.clip_revision,local_files_only=True); clip=CLIPModel.from_pretrained(args.clip_model,revision=args.clip_revision,local_files_only=True).to(args.device); clip.eval()
 except Exception as exc:
  prompt_payload['depression_embedding_sha256']=None; prompt_payload['neutral_embedding_sha256']=None; atomic(out/'semantic_prompt_bank.json',prompt_payload)
  atomic(out/'semantic_search.json',{'version':'ramps-r2-semantic-v1','blocked':True,'reason':'local CLIP processor/model unavailable; no download attempted','clip_model':args.clip_model,'clip_revision':args.clip_revision,'prompt_bank':prompt_payload,'audio_checkpoint_sha256':ah,'video_checkpoint_sha256':vh,'no_test_statement':'No Test rows or metrics were iterated or inspected.'})
  atomic(out/'audit.json',{'version':'ramps-r2-semantic-v1','blocked':True,'reason':'local CLIP processor/model unavailable; no download attempted','train_inference_ran':False,'train_missing_targets_published':False,'no_missing_label_correctness_claim':True,'no_test_statement':'No Test rows or metrics were iterated or inspected.'})
  raise RuntimeError('semantic R2 blocked: local CLIP processor/model unavailable') from exc
 for q in list(audio.parameters())+list(video.parameters())+list(clip.parameters()): q.requires_grad_(False)
 with torch.inference_mode():
  def text_bank(prompts):
   tok=processor(text=prompts,return_tensors='pt',padding=True).to(args.device); return normalize_prompt_bank(clip.get_text_features(**tok)).cpu()
  dep_text,neu_text=text_bank(PROMPTS['depression']),text_bank(PROMPTS['neutral'])
 prompt_payload['depression_embedding_sha256']=semantic_hash(dep_text); prompt_payload['neutral_embedding_sha256']=semantic_hash(neu_text); atomic(out/'semantic_prompt_bank.json',prompt_payload)
 data=WSMAVFusionDataModule(data_root=args.data_root,audio_feature_cache_root=args.audio_feature_cache_root,video_cache_root=args.video_cache_root,batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=True,persistent_workers=False,shuffle_train=False)
 if data.audit['counts']!=EXPECTED: raise RuntimeError('canonical counts failed')
 device=torch.device(args.device); audio.to(device); video.to(device)
 dev=infer(audio,video,clip,dep_text.to(device),neu_text.to(device),data.val_dataset,data.collate_fn,device,args.batch_size,args.num_workers)
 ma=compute_sparse_two_task_metrics(dev['audio_logits'],dev['targets'],dev['observed'],'dev',TASKS); mv=compute_sparse_two_task_metrics(dev['video_logits'],dev['targets'],dev['observed'],'dev',TASKS)
 ga={k:ma[f'dev/{k}/score'] for k in ('depression','parkinson')}; gv={k:mv[f'dev/{k}/score'] for k in ('depression','parkinson')}; ga['mean_score']=ma['dev/mean_score']; gv['mean_score']=mv['dev/mean_score']
 if any(abs(gv[k]-e)>.0005 for k,e in {'depression':.6201013364,'parkinson':.7930427585,'mean_score':.7065720475}.items()): raise RuntimeError(f'video reproduction failed {gv}')
 if any(abs(ga[k]-e)>.0005 for k,e in {'depression':.7479183895,'parkinson':.8277353635,'mean_score':.7878268765}.items()): raise RuntimeError(f'audio reproduction failed {ga}')
 # Frozen Parkinson R2 rules only.
 pidx=torch.where(dev['observed'][:,1])[0]; ps=fit_stats(dev,pidx,1); pr=records(dev,pidx,1,ps); park_rules={'positive':{'enabled':True,'family':'A','tau_conf':.77},'negative':{'enabled':True,'family':'A','tau_conf':.50}}
 for side,pos,exp_sup,exp_prec in (('positive',True,21,1.),('negative',False,193,.9585492)):
  mask=((pr['audio_prob']>=.5)==(pr['video_prob']>=.5))&((pr['audio_prob']>=.5) if pos else (pr['audio_prob']<.5))&(torch.minimum(semantic_confidence(pr['audio_prob'],pos),semantic_confidence(pr['video_prob'],pos))>=park_rules[side]['tau_conf']); sup=int(mask.sum()); prec=float((mask&(pr['labels']==int(pos))).sum()/sup) if sup else 0.; park_rules[side].update({'support':sup,'precision':prec});
  if sup!=exp_sup or abs(prec-exp_prec)>1e-6: raise RuntimeError(f'Parkinson frozen rule mismatch {side}: {sup} {prec}')
 # Depression OOF.
 d_idx=torch.where(dev['observed'][:,0])[0]; labels=dev['targets'][d_idx,0].long(); folds=stratified_folds('depression-semantic',[dev['segment_ids'][int(i)] for i in d_idx],labels,5); oof={k:torch.zeros(len(d_idx),dtype=torch.float64) for k in ('audio_prob','video_prob','semantic_prob','audio_entropy','video_entropy','semantic_entropy','ood_percentile_positive','ood_percentile_negative')}; oof['labels']=labels; fold_audit=[]
 for k in range(5):
  hold=folds==k; fit=~hold; fi=d_idx[fit]; hi=d_idx[hold]; st=fit_stats(dev,fi,0); rr=records(dev,hi,0,st); pos=hold.nonzero().flatten()
  for key in oof:
   if key!='labels': oof[key][pos]=rr[key]
  fold_audit.append({'fold':k,'fit_rows':int(fit.sum()),'heldout_rows':int(hold.sum()),'fit_class_0':int((labels[fit]==0).sum()),'fit_class_1':int((labels[fit]==1).sum()),'audio_temperature':st[0],'video_temperature':st[1],'semantic_calibrator':st[2],'semantic_metrics':semantic_metrics(dev['margin'][fi],dev['targets'][fi,0],st[2])})
 dep_rules,dep_audit=search_semantic_rules(oof,args.precision_target,args.min_support); full_st=fit_stats(dev,d_idx,0); full=records(dev,d_idx,0,full_st); confirmed={}
 for side,pos in (('positive',True),('negative',False)):
  rule=dep_rules[side]
  if not rule.get('enabled',False): confirmed[side]={'enabled':False}; continue
  mask=apply_semantic_rule(full,rule,pos); sup=int(mask.sum()); cor=int((mask&(full['labels']==int(pos))).sum()); confirmed[side]={**rule,'support_full_dev':sup,'precision_full_dev':cor/sup if sup else 0.,'coverage_full_dev':cor/int((full['labels']==int(pos)).sum()),'enabled':sup>=args.min_support and sup>0 and cor/sup>=args.precision_target}
 search={'version':'ramps-r2-semantic-v1','generated_utc':datetime.now(timezone.utc).isoformat(),'audio_checkpoint_path':str(ac.resolve()),'audio_checkpoint_sha256':ah,'video_checkpoint_path':str(vc.resolve()),'video_checkpoint_sha256':vh,'clip_model':args.clip_model,'clip_revision':args.clip_revision,'prompt_bank':prompt_payload,'counts':data.audit['counts'],'no_test_statement':'Only train_dataset and val_dataset may be iterated; no Test rows or metrics were used.','historical_dev_reproduction':{'audio':ga,'video':gv},'frozen_parkinson_rules':park_rules,'depression_folds':fold_audit,'depression_oof_candidate_audit':dep_audit,'depression_selected_oof_rules':dep_rules,'depression_full_dev_audio_temperature':full_st[0],'depression_full_dev_video_temperature':full_st[1],'depression_full_dev_semantic_calibrator':full_st[2],'depression_full_dev_semantic_metrics':semantic_metrics(dev['margin'][d_idx],dev['targets'][d_idx,0],full_st[2]),'depression_full_dev_confirmation':confirmed,'depression_deployable':any(x.get('enabled',False) for x in confirmed.values()),'overall_two_head_deployable':any(x.get('enabled',False) for x in confirmed.values()) and all(x.get('enabled',False) for x in park_rules.values())}
 atomic(out/'semantic_search.json',search)
 if not search['depression_deployable']:
  audit={'version':'ramps-r2-semantic-v1','blocked':True,'reason':'depression semantic reliability gate failed','train_inference_ran':False,'train_missing_targets_published':False,'no_missing_label_correctness_claim':True,'counts':data.audit['counts'],'no_test_statement':'No Test rows or metrics were iterated or inspected.','depression':confirmed,'parkinson_frozen_rules':park_rules,'prompt_bank_sha256':prompt_payload['canonical_json_sha256']}; atomic(out/'audit.json',audit); raise RuntimeError('R2 semantic blocked: depression has no deployable side')
 # The deployment path is reachable only after the two-head DEV gate.
 train=infer(audio,video,clip,dep_text.to(device),neu_text.to(device),data.train_dataset,data.collate_fn,device,args.batch_size,args.num_workers)
 if len(train['segment_ids'])!=6325 or len(set(train['segment_ids']))!=6325: raise RuntimeError('TRAIN canonical rows/IDs invalid')
 pseudo_accept=torch.zeros((6325,2),dtype=torch.bool); pseudo_targets=torch.full((6325,2),float('nan'),dtype=torch.float64); pseudo_reliability=torch.zeros((6325,2),dtype=torch.float64); pseudo_class=torch.full((6325,2),-1,dtype=torch.long)
 dep_train=records(train,torch.arange(6325),0,full_st)
 for side,pos in (('positive',True),('negative',False)):
  rule=confirmed[side]; mask=apply_semantic_rule(dep_train,rule,pos)&~train['observed'][:,0]
  pseudo_accept[:,0]|=mask; pseudo_targets[mask,0]=dep_train['audio_prob'][mask]; pseudo_reliability[mask,0]=semantic_reliability(dep_train,rule,pos,mask)[mask]; pseudo_class[mask,0]=int(pos)
 park_train=records(train,torch.arange(6325),1,ps)
 for side,pos in (('positive',True),('negative',False)):
  rule=park_rules[side]; mask=(((park_train['audio_prob']>=.5)==(park_train['video_prob']>=.5))&((park_train['audio_prob']>=.5) if pos else (park_train['audio_prob']<.5))&(torch.minimum(semantic_confidence(park_train['audio_prob'],pos),semantic_confidence(park_train['video_prob'],pos))>=rule['tau_conf'])&~train['observed'][:,1])
  pseudo_accept[:,1]|=mask; pseudo_targets[mask,1]=park_train['audio_prob'][mask]; pseudo_reliability[mask,1]=torch.minimum(semantic_confidence(park_train['audio_prob'],pos),semantic_confidence(park_train['video_prob'],pos)).detach()[mask].clamp(0,1); pseudo_class[mask,1]=int(pos)
 observed_overwrite=int((pseudo_accept&train['observed']).sum()); pseudo_observed=int(torch.isfinite(pseudo_targets[train['observed']]).sum()); accepted=pseudo_targets[pseudo_accept]
 if observed_overwrite or pseudo_observed or (accepted.numel() and not bool(torch.isfinite(accepted).all())) or (accepted.numel() and not bool(((accepted>=0)&(accepted<=1)).all())): raise RuntimeError('TRAIN pseudo-target invariant failed')
 if not bool(pseudo_accept[~train['observed']].any(dim=0).all()): raise RuntimeError('TRAIN two-head acceptance gate failed')
 artifact={'version':'ramps-r2-semantic-v1','task_names':list(TASKS),'segment_ids':train['segment_ids'],'observed_mask':train['observed'],'observed_targets':train['targets'],'raw_audio_logits':train['audio_logits'],'calibrated_audio_probs':torch.stack([dep_train['audio_prob'],park_train['audio_prob']],1),'raw_video_logits':train['video_logits'],'calibrated_video_probs':torch.stack([dep_train['video_prob'],park_train['video_prob']],1),'semantic_margin_depression':train['margin'],'calibrated_semantic_prob_depression':dep_train['semantic_prob'].float(),'audio_task_features':train['audio_features'],'pseudo_accept_mask':pseudo_accept,'pseudo_targets':pseudo_targets,'pseudo_reliability':pseudo_reliability,'pseudo_class':pseudo_class,'depression_selected_rules':confirmed,'parkinson_frozen_rules':park_rules,'audio_temperatures':{'depression':full_st[0],'parkinson':ps[0]},'video_temperatures':{'depression':full_st[1],'parkinson':ps[1]},'semantic_calibrator':full_st[2],'semantic_prompt_bank':prompt_payload,'semantic_prompt_bank_sha256':prompt_payload['canonical_json_sha256'],'clip_model_name':args.clip_model,'clip_model_revision':args.clip_revision,'audio_checkpoint_path':str(ac.resolve()),'audio_checkpoint_sha256':ah,'video_checkpoint_path':str(vc.resolve()),'video_checkpoint_sha256':vh}
 atomic_torch(out/'train_missing_targets.pt',artifact)
 audit={'version':'ramps-r2-semantic-v1','blocked':False,'train_inference_ran':True,'train_missing_targets_published':True,'no_missing_label_correctness_claim':True,'counts':data.audit['counts'],'train_rows':6325,'observed_overwrite_violations':observed_overwrite,'pseudo_values_on_observed_entries':pseudo_observed,'accepted_missing_entries':int(pseudo_accept.sum()),'tasks':{}}
 for task,name in enumerate(TASKS):
  missing=~train['observed'][:,task]; acc=pseudo_accept[:,task]; vals=pseudo_targets[acc,task]; rel=pseudo_reliability[acc,task]; audit['tasks'][name]={'missing_train_count':int(missing.sum()),'accepted_total':int(acc.sum()),'rejected_total':int((missing&~acc).sum()),'coverage':float(acc.sum()/missing.sum()),'accepted_positive_count':int((acc&(pseudo_class[:,task]==1)).sum()),'accepted_negative_count':int((acc&(pseudo_class[:,task]==0)).sum()),'accepted_probability_stats':stats(vals),'reliability_stats':stats(rel)}
 atomic(out/'audit.json',audit)
 print(json.dumps({'status':'passed','accepted_missing':int(pseudo_accept.sum()),'output_root':str(out)},sort_keys=True))
if __name__=='__main__': main()
