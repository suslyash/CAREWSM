#!/usr/bin/env python3
"""Compare one saved extraction per model/fold with CUDA-AMP checkpoint forward."""
from __future__ import annotations
import csv, json, pickle
from pathlib import Path
import numpy as np
import torch
import chimera_plugin
from chimera_ml.core.config import ExperimentConfig
from chimera_ml.training.builders import BuildContext, build_datamodule, build_model

PAIRS={'id1':('deep','interp_B_mean_gated64_equal'),'id2':('deep_phon','interp_BP_mean_gated64_equal'),'id4':('deep_egemaps_phon_artic','interp_BEPA_mean_gated64_equal')}

def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required to reproduce extraction AMP')
    ledger={(r['experiment_id'],int(r['fold'])):r['checkpoint_path'] for r in csv.DictReader(open('reports/audio_interpretable_feature_fusion/provenance_ledger.csv'))}
    chimera_plugin.register(); results=[]
    for model_id,(prefix,experiment) in PAIRS.items():
      for fold in range(5):
        cfg=ExperimentConfig.from_yaml(f'configs/audio_feature_extraction/{prefix}_fold{fold}.yaml'); ctx=BuildContext(config=cfg,stage='eval')
        dm=build_datamodule(cfg.section('data'),context=ctx); ctx.register(dm); model=build_model(cfg.section('model'),context=ctx)
        state=torch.load(ledger[(experiment,fold)],map_location='cpu',weights_only=True); model.load_state_dict(state['model_state_dict']); model.cuda().eval()
        batch=next(iter(dm.train_dataloader()['train_care'])); meta=batch.meta['sample_meta'][0]
        def move(value): return value.cuda(non_blocking=True) if isinstance(value,torch.Tensor) else value
        batch.inputs={k:move(v) for k,v in batch.inputs.items()}; batch.targets=move(batch.targets); batch.masks={k:move(v) for k,v in batch.masks.items()}
        with torch.inference_mode(), torch.amp.autocast('cuda',enabled=True): out=model(batch)
        root=Path(cfg.section('data')['params']['care_root']); path=root/'extracted_unimodal'/meta['label_id']/meta['subgroup']/f"PROFILE_{str(meta['profile_id']).removeprefix('PROFILE_')}"/f"VIDEO_{str(meta['video_id']).removeprefix('VIDEO_')}"/f'audio_{model_id}_fold{fold}.pickle'
        with path.open('rb') as h: saved=pickle.load(h)
        values=out.aux['features'][0].float().cpu().numpy(); logits=out.preds[0].float().cpu().numpy()
        diffs={'shared':float(np.max(np.abs(saved['shared_features']-values[None,:64]))),'depression':float(np.max(np.abs(saved['depression_features']-values[None,64:128]))),'pd':float(np.max(np.abs(saved['pd_features']-values[None,128:]))),'logits':float(np.max(np.abs(saved['logits']-logits)))}
        results.append({'model':model_id,'fold':fold,**diffs})
        print(json.dumps(results[-1],sort_keys=True))
    Path('/tmp/carewsm-feature-audit-114').mkdir(exist_ok=True)
    Path('/tmp/carewsm-feature-audit-114/forward_amp.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__': main()
