"""Deterministic, dependency-light audit of extracted audio feature artifacts."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, time, ast, zipfile, re, struct
from collections import Counter, defaultdict
from pathlib import Path

STATUSES = ("VALID", "MISSING", "ZERO_BYTE", "UNREADABLE", "SCHEMA_MISMATCH", "EMPTY", "NONFINITE", "INVALID_DTYPE", "INVALID_SHAPE", "IDENTITY_MISMATCH", "OTHER_ERROR")

def _read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))

def _finite(value):
    try: return math.isfinite(float(value))
    except (TypeError, ValueError): return False

def _npy_header(data):
    if not data.startswith(b"\x93NUMPY"): raise ValueError("not npy")
    major=data[6]; header_len=int.from_bytes(data[8:10], "little") if major==1 else int.from_bytes(data[8:12], "little")
    header=data[10:10+header_len] if major==1 else data[12:12+header_len]
    return ast.literal_eval(header.decode("latin1").strip()), (10+header_len if major==1 else 12+header_len)

def validate_artifact(path: Path, spec: dict) -> tuple[str, str]:
    if not path.exists(): return "MISSING", "file does not exist"
    if path.stat().st_size == 0: return "ZERO_BYTE", "zero-byte file"
    try:
        if spec["format"] == "csv":
            with path.open(newline="", encoding="utf-8-sig") as f:
                r=csv.DictReader(f); cols=r.fieldnames or []; required=list(spec.get("metadata", []))
                if not set(required).issubset(cols): return "SCHEMA_MISMATCH", "missing metadata columns"
                rows=list(r)
            if not rows: return "EMPTY", "no data rows"
            numeric=[c for c in cols if c not in required]
            if not numeric: return "SCHEMA_MISMATCH", "no feature columns"
            for row in rows:
                for c in numeric:
                    if row.get(c,"")=="": return "INVALID_DTYPE", "blank/non-numeric feature value"
                    try: value=float(row[c])
                    except (TypeError, ValueError): return "INVALID_DTYPE", "non-numeric feature value"
                    if not math.isfinite(value): return "NONFINITE", "non-finite numeric value"
            for field in ("frame_start_sec","turn_start_sec"):
                if field in cols:
                    vals=[float(x[field]) for x in rows if x.get(field,"")!=""]
                    if any(b<a for a,b in zip(vals,vals[1:])): return "INVALID_SHAPE", "timestamps are not monotonic"
            return "VALID", f"rows={len(rows)} cols={len(cols)}"
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.endswith('.npy')]
            if not names: return "EMPTY", "no npy members"
            required={k+'.npy' for k in spec.get('metadata', [])}
            if not required.issubset(set(names)): return "SCHEMA_MISMATCH", "missing npz keys"
            details=[]
            for name in names:
                raw=z.read(name); h,offset=_npy_header(raw[:512]); shape=h.get('shape',()); descr=h.get('descr','')
                if not shape or any(int(x)==0 for x in shape): return "EMPTY", f"empty array: {name}"
                kind=(descr[-2:-1] if len(descr)>1 else descr)
                if kind not in 'biufc': return "INVALID_DTYPE", f"non-numeric array: {name}"
                if kind in 'fc':
                    count=1
                    for dim in shape: count*=int(dim)
                    size=int(re.search(r'(\d+)$',descr).group(1))
                    endian='>' if descr.startswith('>') else '<'
                    fmt=endian+('f' if kind=='f' and size==4 else 'd' if kind=='f' else 'e')
                    try: values=struct.unpack(fmt[0]+fmt[1:]*count,raw[offset:offset+size*count])
                    except (struct.error,TypeError): return "INVALID_SHAPE", f"data length mismatch: {name}"
                    if any(not math.isfinite(v) for v in values): return "NONFINITE", f"non-finite value: {name}"
                details.append(f"{name}:{shape}:{descr}")
            return "VALID", ";".join(details)
    except (OSError, ValueError, EOFError, UnicodeError, zipfile.BadZipFile, KeyError) as e:
        return "UNREADABLE", type(e).__name__

def _feature_specs(registry):
    return {k: v for k, v in registry["sources"].items() if v.get("modality") == "audio"}

def _artifact(root: Path, row: dict, filename: str) -> Path:
    # Split video_dir values use legacy storage paths; extracted is authoritative.
    label=row.get("label_id", "") or row.get("stratum", "")
    subgroup=row.get("subgroup", "")
    profile=row.get("profile_id", "")
    video=row.get("video_id", "")
    profile_dir=profile if profile.startswith("PROFILE_") else "PROFILE_"+profile
    candidate=root/"extracted"/label/subgroup/profile_dir/(video if video.startswith("VIDEO_") else "VIDEO_"+video)
    if candidate.exists(): return candidate/filename
    rel=row.get("video_dir", ""); parts=Path(rel).parts
    if parts and parts[0] in (root.name,"CARE","WSM_NEW"): rel=Path(*parts[1:])
    return root/rel/filename

def audit(corpus_roots: dict[str, Path], registry_path: Path, out: Path) -> dict:
    started = time.monotonic(); registry = json.loads(registry_path.read_text()); specs = _feature_specs(registry)
    manifest=[]; statuses=Counter(); coverage=Counter(); split_notes=[]; profile_splits={}; split_rows_by={}
    for corpus, root in corpus_roots.items():
        split_dir=root/"splits"; files={n: split_dir/f"{n}.csv" for n in ("videos","train","dev","test")}
        rows_by={n: (_read_csv(p) if p.exists() else []) for n,p in files.items()}
        split_rows_by[corpus]=rows_by
        if not files["videos"].exists(): split_notes.append(f"{corpus}: missing splits/videos.csv")
        all_rows=[]
        for n in ("train","dev","test"):
            for row in rows_by[n]: row=dict(row); row["_declared_file"]=n; all_rows.append(row)
        video_index={r.get("sample_id"):r for r in rows_by["videos"]}
        for row in all_rows:
            sample=row.get("sample_id",""); labels={"depression": "", "parkinson": ""}
            if corpus=="WSM" and rows_by["videos"]:
                source=video_index.get(sample, {})
                labels={"depression":source.get("y_depression", ""),"parkinson":source.get("y_parkinson", "")}
            else:
                labels["depression"] = "1" if row.get("label_id","").upper()=="DEPRESSION" else ("0" if row.get("label_id","").upper() in ("CONTROL","PARKINSON") else "")
                labels["parkinson"] = "1" if row.get("label_id","").upper()=="PARKINSON" else ("0" if row.get("label_id","").upper() in ("CONTROL","DEPRESSION") else "")
            rec={"corpus":corpus,"sample_id":sample,"profile_id":row.get("profile_id",""),"video_id":row.get("video_id",""),"split":row.get("split",row.get("_declared_file","")),"declared_file":row.get("_declared_file",""),"cv_fold":row.get("cv_fold",""),"depression":labels["depression"],"depression_observed":bool(labels["depression"]!=""),"parkinson":labels["parkinson"],"parkinson_observed":bool(labels["parkinson"]!="")}
            for name,spec in specs.items():
                status,reason=validate_artifact(_artifact(root,row,spec["file"]),spec); rec[name+"_status"]=status; rec[name+"_valid"]=status=="VALID"; statuses[(corpus,name,status)]+=1
            manifest.append(rec)
        for n in ("train","dev","test"):
            if any(r.get("split") != n for r in rows_by[n]): split_notes.append(f"{corpus}: {n}.csv contains rows with split != {n}")
    out.mkdir(parents=True, exist_ok=True)
    fields=list(manifest[0]) if manifest else []
    mp=out/"sample_feature_availability.csv"
    with mp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(manifest)
    digest=hashlib.sha256(mp.read_bytes()).hexdigest()
    valid_cols=[n+"_valid" for n in specs]
    pairwise={}
    for i,a in enumerate(specs):
        for b in list(specs)[i:]:
            pairwise[a+"__"+b]=sum(bool(r[a+"_valid"] and r[b+"_valid"]) for r in manifest)
    complete_case=sum(all(r[c] for c in valid_cols) for r in manifest)
    label_coverage={}
    for corpus in corpus_roots:
        subset=[r for r in manifest if r["corpus"]==corpus]
        label_coverage[corpus]={t:{"observed":sum(r[t+"_observed"] for r in subset),"positive":sum(r[t]=="1" for r in subset if r[t+"_observed"]),"negative":sum(r[t]=="0" for r in subset if r[t+"_observed"])} for t in ("depression","parkinson")}
    split_contract={}
    for corpus,root in corpus_roots.items():
        split_contract[corpus]={}; seen={}
        for n in ("train","dev","test"):
            rows=split_rows_by[corpus].get(n,[]); ids=[r.get("sample_id","") for r in rows]; profiles={r.get("profile_id","") for r in rows}
            seen[n]=set(ids); split_contract[corpus][n]={"rows":len(rows),"profiles":len(profiles),"duplicate_sample_ids":len(ids)-len(set(ids)),"declared_split_mismatch":sum(r.get("split")!=n for r in rows)}
        for a in ("train","dev","test"):
            for b in ("train","dev","test"):
                if a<b:
                    split_contract[corpus][a+"_vs_"+b+"_sample_overlap"]=len(seen[a]&seen[b])
                    split_contract[corpus][a+"_vs_"+b+"_profile_overlap"]=len({r.get("profile_id","") for r in split_rows_by[corpus].get(a,[])} & {r.get("profile_id","") for r in split_rows_by[corpus].get(b,[])})
        folds={}
        for k in range(5):
            tr=_read_csv(root/"splits"/"folds"/f"fold_{k}_train.csv") if (root/"splits"/"folds"/f"fold_{k}_train.csv").exists() else []
            va=_read_csv(root/"splits"/"folds"/f"fold_{k}_val.csv") if (root/"splits"/"folds"/f"fold_{k}_val.csv").exists() else []
            tr_ids={x.get("sample_id","") for x in tr}; va_ids={x.get("sample_id","") for x in va}; tr_profiles={x.get("profile_id","") for x in tr}; va_profiles={x.get("profile_id","") for x in va}
            folds[str(k)]={"train_rows":len(tr),"val_rows":len(va),"train_profiles":len(tr_profiles),"val_profiles":len(va_profiles),"sample_overlap":len(tr_ids&va_ids),"profile_overlap":len(tr_profiles&va_profiles),"train_sample_ids":sorted(tr_ids),"val_sample_ids":sorted(va_ids),"train_profiles_set":sorted(tr_profiles),"val_profiles_set":sorted(va_profiles)}
        split_contract[corpus]["folds"]=folds
        for a in range(5):
            for b in range(a+1,5):
                fa,fb=folds[str(a)],folds[str(b)]
                split_contract[corpus][f"fold_{a}_vs_{b}"]={"val_sample_overlap":len(set(fa["val_sample_ids"])&set(fb["val_sample_ids"])),"val_profile_overlap":len(set(fa["val_profiles_set"])&set(fb["val_profiles_set"])),"train_sample_overlap":len(set(fa["train_sample_ids"])&set(fb["train_sample_ids"]))}
        for f in folds.values():
            for key in ("train_sample_ids","val_sample_ids","train_profiles_set","val_profiles_set"): f.pop(key,None)
    joint={}
    for k in range(5):
        joint[str(k)]={}
        for corpus in corpus_roots:
            f=split_contract[corpus]["folds"][str(k)]; joint[str(k)][corpus]={"train_rows":f["train_rows"],"val_rows":f["val_rows"],"train_profiles":f["train_profiles"],"val_profiles":f["val_profiles"]}
        joint[str(k)]["train_rows_total"]=sum(joint[str(k)][c]["train_rows"] for c in corpus_roots); joint[str(k)]["val_rows_total"]=sum(joint[str(k)][c]["val_rows"] for c in corpus_roots)
    report={"runtime_sec":round(time.monotonic()-started,3),"rows":len(manifest),"feature_families":list(specs),"statuses":{f"{a}|{b}|{c}":n for (a,b,c),n in sorted(statuses.items())},"manifest":{"path":str(mp.resolve()),"sha256":digest,"bytes":mp.stat().st_size,"rows":len(manifest)},"split_notes":split_notes,"split_contract":split_contract,"joint_fold_compatibility":joint,"pairwise_joint_valid":pairwise,"complete_case_all_families":complete_case,"complete_case_percent":100*complete_case/len(manifest) if manifest else 0,"label_coverage":label_coverage}
    (out/"audit_summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report

def main():
    p=argparse.ArgumentParser(); p.add_argument("--care-root",type=Path,default=Path("/media/maxim/Databases/CARE")); p.add_argument("--wsm-root",type=Path,default=Path("/media/maxim/Databases/WSM_NEW")); p.add_argument("--registry",type=Path,default=Path("/media/maxim/Databases/CARE/splits/features.json")); p.add_argument("--out",type=Path,default=Path("reports/feature_audit")); a=p.parse_args(); print(json.dumps(audit({"CARE":a.care_root,"WSM":a.wsm_root},a.registry,a.out),indent=2))
if __name__ == "__main__": main()
