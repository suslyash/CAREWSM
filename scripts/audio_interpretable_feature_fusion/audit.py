#!/usr/bin/env python3
"""Audit seven-family availability and write privacy-safe provenance artifacts."""
import csv, hashlib, json
from collections import Counter
from pathlib import Path

FAMILIES=("emotionmsp_embeds","emotionmsp_trajectories","trillsson4_1","wav2vec_xlsr300m","egemaps_func","disvoice_phonation_static","disvoice_articulation_static")
MANIFEST=Path("reports/feature_audit/sample_feature_availability.csv")
OUT=Path("reports/audio_interpretable_feature_fusion")

def digest(value): return hashlib.sha256(value.encode()).hexdigest()[:20]
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open(newline='',encoding='utf-8') as h: rows=list(csv.DictReader(h))
    complete=[r for r in rows if all(r.get(f+"_status")=="VALID" for f in FAMILIES)]
    counts={}
    for corpus in ("CARE","WSM"):
        for split in ("train","dev","test"):
            subset=[r for r in complete if r.get("corpus")==corpus and r.get("split")==split]
            counts[f"{corpus}_{split}"]={"n":len(subset),"depression_observed":sum(r.get("depression_observed")=="True" for r in subset),"parkinson_observed":sum(r.get("parkinson_observed")=="True" for r in subset),"depression_positive":sum(r.get("depression")=="1" for r in subset),"parkinson_positive":sum(r.get("parkinson")=="1" for r in subset)}
    dictionary={"families":[],"cohort_policy":"complete-case across all seven families","aggregation":"independent per-family arithmetic mean over native turns; concatenation in fixed family order","family_order":list(FAMILIES)}
    with Path('/media/maxim/Databases/CARE/splits/features.json').open(encoding='utf-8') as h: registry=json.load(h)["sources"]
    for i,family in enumerate(FAMILIES):
        spec=registry[family]
        dictionary["families"].append({"name":family,"input_index_start":sum(int(registry[x].get("original_feature_dim",0)) for x in FAMILIES[:i]),"dimension":spec.get("original_feature_dim"),"level":spec.get("level"),"format":spec.get("format"),"metadata":spec.get("metadata",[]),"source_file":spec.get("file")})
    (OUT/"feature_dictionary.json").write_text(json.dumps(dictionary,indent=2)+"\n",encoding='utf-8')
    with (OUT/"complete_case_sample_manifest.csv").open('w',newline='',encoding='utf-8') as h:
        fields=['corpus','split','fold','sample_hash','profile_hash','feature_families']
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for r in complete:
            w.writerow({'corpus':r.get('corpus',''),'split':r.get('split',''),'fold':r.get('cv_fold',''),'sample_hash':digest(r.get('sample_id','')),'profile_hash':digest(r.get('profile_id','')),'feature_families':'+'.join(FAMILIES)})
    result={"manifest":str(MANIFEST),"families":list(FAMILIES),"manifest_rows":len(rows),"complete_case_rows":len(complete),"counts":counts,"status":"PASS"}
    (OUT/"cohort_audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
