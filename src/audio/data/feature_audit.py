"""Deterministic, dependency-light audit of extracted audio feature artifacts."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, time
from collections import Counter, defaultdict
from pathlib import Path

STATUSES = ("VALID", "MISSING", "ZERO_BYTE", "UNREADABLE", "SCHEMA_MISMATCH", "EMPTY", "NONFINITE", "INVALID_DTYPE", "INVALID_SHAPE", "IDENTITY_MISMATCH", "OTHER_ERROR")

def _read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))

def _finite(value):
    try: return math.isfinite(float(value))
    except (TypeError, ValueError): return False

def validate_artifact(path: Path, spec: dict) -> tuple[str, str]:
    if not path.exists(): return "MISSING", "file does not exist"
    if path.stat().st_size == 0: return "ZERO_BYTE", "zero-byte file"
    try:
        if spec["format"] == "csv":
            with path.open(newline="", encoding="utf-8-sig") as f:
                r = csv.DictReader(f); cols = r.fieldnames or []
                required = list(spec.get("metadata", []))
                if not set(required).issubset(cols): return "SCHEMA_MISMATCH", "missing metadata columns"
                rows = list(r)
            if not rows: return "EMPTY", "no data rows"
            numeric = [c for c in cols if c not in required]
            if not numeric: return "SCHEMA_MISMATCH", "no feature columns"
            for row in rows:
                for c in numeric:
                    if row.get(c, "") == "": return "INVALID_DTYPE", "blank/non-numeric feature value"
                    if not _finite(row[c]): return "NONFINITE", "non-finite numeric value"
            times = [float(row["frame_start_sec"]) for row in rows if row.get("frame_start_sec", "") != ""]
            if times and any(b < a for a, b in zip(times, times[1:])): return "INVALID_SHAPE", "timestamps are not monotonic"
            return "VALID", f"rows={len(rows)} cols={len(cols)}"
        import numpy as np
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.files); required = set(spec.get("metadata", []))
            if not required.issubset(keys): return "SCHEMA_MISMATCH", "missing npz keys"
            arrays = [z[k] for k in z.files if k not in required]
            if not arrays: return "EMPTY", "no embedding arrays"
            if any(a.size == 0 for a in arrays): return "EMPTY", "empty array"
            if any(not np.issubdtype(a.dtype, np.number) for a in arrays): return "INVALID_DTYPE", "non-numeric array"
            if any(not np.isfinite(a).all() for a in arrays): return "NONFINITE", "non-finite array"
            if any(a.ndim < 1 for a in arrays): return "INVALID_SHAPE", "scalar array"
            return "VALID", "keys=" + ",".join(sorted(keys))
    except (OSError, ValueError, EOFError, UnicodeError, Exception) as e:
        return "UNREADABLE", type(e).__name__

def _feature_specs(registry):
    return {k: v for k, v in registry["sources"].items() if v.get("modality") == "audio"}

def _artifact(root: Path, row: dict, filename: str) -> Path:
    rel = row.get("video_dir", "")
    parts = Path(rel).parts
    if parts and parts[0] in (root.name, "CARE", "WSM_NEW"): rel = Path(*parts[1:])
    return root / rel / filename

def audit(corpus_roots: dict[str, Path], registry_path: Path, out: Path) -> dict:
    started = time.monotonic(); registry = json.loads(registry_path.read_text()); specs = _feature_specs(registry)
    manifest=[]; statuses=Counter(); coverage=Counter(); split_notes=[]; profile_splits={}
    for corpus, root in corpus_roots.items():
        split_dir=root/"splits"; files={n: split_dir/f"{n}.csv" for n in ("videos","train","dev","test")}
        rows_by={n: (_read_csv(p) if p.exists() else []) for n,p in files.items()}
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
    report={"runtime_sec":round(time.monotonic()-started,3),"rows":len(manifest),"feature_families":list(specs),"statuses":{f"{a}|{b}|{c}":n for (a,b,c),n in sorted(statuses.items())},"manifest":{"path":str(mp.resolve()),"sha256":digest,"bytes":mp.stat().st_size,"rows":len(manifest)},"split_notes":split_notes}
    (out/"audit_summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report

def main():
    p=argparse.ArgumentParser(); p.add_argument("--care-root",type=Path,default=Path("/media/maxim/Databases/CARE")); p.add_argument("--wsm-root",type=Path,default=Path("/media/maxim/Databases/WSM_NEW")); p.add_argument("--registry",type=Path,default=Path("/media/maxim/Databases/CARE/splits/features.json")); p.add_argument("--out",type=Path,default=Path("reports/feature_audit")); a=p.parse_args(); print(json.dumps(audit({"CARE":a.care_root,"WSM":a.wsm_root},a.registry,a.out),indent=2))
if __name__ == "__main__": main()
