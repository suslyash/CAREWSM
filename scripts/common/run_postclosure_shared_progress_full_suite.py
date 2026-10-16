#!/usr/bin/env python3
"""TASK-011B deterministic DEV-only suite orchestration."""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, sqlite3, statistics, subprocess, sys
from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml

REPO = Path(__file__).resolve().parents[2]
SUITE = REPO / "configs/wsm_mm_pd_dep_v1/postclosure_shared_progress_suite"
MANIFEST = SUITE / "suite_manifest.json"
LEDGER = REPO / "logs/task011b_shared_progress_suite/execution_ledger.json"
RESULTS = REPO / "logs/task011b_shared_progress_suite/dev_only_results.json"
REPORT = REPO / "docs/POSTCLOSURE_SHARED_PROGRESS_FULL_SUITE_EN.md"
PARENT = REPO / "logs/wsm_mm_pd_dep_v1/postclosure_shared_progress_optuna_seed42_2026-10-08_15-14_wsm_av_r3_disease_query_model_task011a-shared-progress-8a53-029_f5dbef29/task011a-shared-progress-8a53-029.yaml"
PSEUDO = Path("/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt")
UNIFORM = Path("/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_reliability_ablation/uniform_accepted_reliability_v1.pt")
NO_SEM = Path("/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_semantic_ablation/no_semantic_depression_v1.pt")
SHUFFLED = Path("/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001_exact_fix1.pt")
PSEUDO_SHA = "17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945"
T4 = 2.7764451051977987

def stop(msg: str) -> None:
    raise SystemExit(msg)

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    def add(group: str, variant: str, seeds: list[int], intervention: str) -> None:
        for seed in seeds:
            out.append({"group":group, "variant":variant, "seed":seed,
                        "intervention":intervention,
                        "run_name":f"task011b_{variant}_seed{seed}"})
    add("A","shared_progress_parent",[43,44,45,46],"new seed confirmation of frozen trial-28 parent")
    for v,i in [("no_direct_pseudo","warmup final_scale=0.0"),
                ("uniform_reliability","uniform accepted reliability cache"),
                ("no_semantic_depression","no-semantic-depression cache"),
                ("no_video_online","modality_available_override=[true,false]"),
                ("no_audio_online","modality_available_override=[false,true]")]:
        add("B",v,[42,43,44],i)
    add("C","task_aware_r4_progress",[42,43,44],"task_aware_fusion=false -> true")
    add("D","shuffled_pseudo",[42,43,44],"accepted deterministic shuffled pseudo cache")
    add("E","depression_only",[42,43,44],"training_task=depression; selector=dev/depression/score")
    add("E","parkinson_only",[42,43,44],"training_task=parkinson; selector=dev/parkinson/score")
    for v,i in [("equal","loss mode=equal"),("static_stch","loss mode=stch; tau=0.1"),
                ("ra_stch","loss mode=ra_stch; frozen controller constants")]:
        add("F",v,[42,43,44],i)
    for v,i in [("gradnorm","gradient method=gradnorm; alpha=1.5; weight_lr=0.025"),
                ("pcgrad","gradient method=pcgrad; seed-bound"),
                ("cagrad","gradient method=cagrad; calpha=0.5; rescale=1"),
                ("dbmtl","gradient method=dbmtl; beta=0.9; beta_sigma=0.0")]:
        add("G",v,[42,43,44],i)
    if len(out) != 52: stop(f"matrix error: {len(out)}")
    return out

def load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict): stop(f"not a mapping: {path}")
    return value

def sha_cache_contract(path: Path) -> dict[str, Any]:
    import torch
    value = torch.load(path, map_location="cpu", weights_only=False)
    accept, observed = value["pseudo_accept_mask"].bool(), value["observed_mask"].bool()
    classes = value["pseudo_class"].long()
    return {"accepted_counts":[int((accept[:,i] & ~observed[:,i]).sum()) for i in range(2)],
            "positive_counts":[int((accept[:,i] & (classes[:,i] == 1)).sum()) for i in range(2)],
            "negative_counts":[int((accept[:,i] & (classes[:,i] == 0)).sum()) for i in range(2)],
            "missing_counts":[int((~observed[:,i]).sum()) for i in range(2)]}

def replace_balancer(c: dict[str, Any], callback: str, params: dict[str, Any] | None = None) -> None:
    c["callbacks"] = [x for x in c["callbacks"]
                      if x.get("name") not in {"wsm_r4_balance_callback","wsm_gradient_mtl_callback"}]
    c["callbacks"].append({"name":callback,"params":params or {}})

def selector(c: dict[str, Any], key: str) -> None:
    for x in c["callbacks"]:
        if x.get("name") in {"checkpoint_callback","early_stopping_callback","sweep_target_callback"}:
            x.setdefault("params", {})["monitor"] = key
            x["params"]["mode"] = "max"

def materialize(parent: dict[str, Any], spec: dict[str, Any]) -> tuple[dict[str, Any],dict[str,Any]]:
    c = deepcopy(parent)
    v, seed = spec["variant"], spec["seed"]
    c["seed"] = seed
    c["experiment_info"]["params"]["run_name"] = spec["run_name"]
    c["data"]["name"] = "wsm_ramps_semantic_dev_only_datamodule"
    dp = c["data"]["params"]
    for k in ("modality_available_override","expected_accepted_counts","expected_positive_counts","expected_negative_counts"):
        dp.pop(k, None)
    cache = PSEUDO
    accepted, positive, negative = [376,1801], [376,212], [0,1589]
    if v == "shared_progress_parent":
        pass
    elif v == "no_direct_pseudo":
        for x in c["callbacks"]:
            if x.get("name") == "wsm_pseudo_scale_warmup_callback": x["params"]["final_scale"] = 0.0
    elif v == "uniform_reliability": cache = UNIFORM
    elif v == "no_semantic_depression": cache, accepted = NO_SEM, [0,1801]
    elif v == "no_video_online": dp["modality_available_override"] = [True,False]
    elif v == "no_audio_online": dp["modality_available_override"] = [False,True]
    elif v == "task_aware_r4_progress": c["model"]["params"]["task_aware_fusion"] = True
    elif v == "shuffled_pseudo": cache = SHUFFLED
    elif v == "depression_only":
        c["loss"]["params"]["training_task"] = "depression"; selector(c,"dev/depression/score")
    elif v == "parkinson_only":
        c["loss"]["params"]["training_task"] = "parkinson"; selector(c,"dev/parkinson/score")
    elif v in {"equal","static_stch","ra_stch"}:
        c["loss"]["params"]["mode"] = {"equal":"equal","static_stch":"stch","ra_stch":"ra_stch"}[v]
        if v in {"static_stch","ra_stch"}: c["loss"]["params"]["tau"] = 0.1
        if v == "ra_stch": c["loss"]["params"].update({"controller_ema":0.8,"grad_ema":0.9,"reliability_ema":0.9,"weight_min":0.2,"weight_max":0.8})
    elif v in {"gradnorm","pcgrad","cagrad","dbmtl"}:
        c["loss"]["name"] = "wsm_gradient_mtl_loss"; lp = c["loss"]["params"]; lp.pop("mode",None)
        lp.update({"method":v,"pseudo_scale":0.0})
        if v == "gradnorm": lp.update({"gradnorm_alpha":1.5,"gradnorm_weight_lr":0.025})
        if v == "pcgrad": lp["pcgrad_seed"] = seed
        if v == "cagrad": lp.update({"calpha":0.5,"rescale":1})
        if v == "dbmtl": lp.update({"db_beta":0.9,"db_beta_sigma":0.0})
        replace_balancer(c,"wsm_gradient_mtl_callback")
    else: stop(f"unknown variant: {v}")
    dp["pseudo_cache_path"] = str(cache)
    if v == "no_semantic_depression":
        dp.update({"expected_accepted_counts":accepted,"expected_positive_counts":[0,212],"expected_negative_counts":[0,1589]})
    meta={"cache_path":str(cache),"cache_sha256":digest(cache),"accepted_counts":accepted,
          "positive_counts":positive,"negative_counts":negative,"missing_counts":[2665,3660],
          "model_mode":"task_aware" if c["model"]["params"].get("task_aware_fusion",True) else "shared",
          "loss_mode":c["loss"]["params"].get("mode"),"loss_method":c["loss"]["params"].get("method"),
          "selector":"dev/depression/score" if v=="depression_only" else "dev/parkinson/score" if v=="parkinson_only" else "dev/mean_score"}
    return c,meta

def prepare() -> None:
    if not PARENT.is_file(): stop(f"parent config missing: {PARENT}")
    for p in (PSEUDO,UNIFORM,NO_SEM,SHUFFLED):
        if not p.is_file(): stop(f"cache missing: {p}")
    parent=load(PARENT)
    if parent.get("seed") != 42 or parent["model"]["params"].get("task_aware_fusion") is not False: stop("parent is not frozen Shared trial-28")
    SUITE.mkdir(parents=True,exist_ok=True)
    for p in SUITE.glob("*.yaml"): p.unlink()
    output=[]
    for n,spec in enumerate(rows(),1):
        c,meta=materialize(parent,spec); path=SUITE/f"{n:02d}_{spec['group']}_{spec['variant']}_seed{spec['seed']}.yaml"
        path.write_text(yaml.safe_dump(c,sort_keys=False))
        item=dict(spec,index=n,config_path=str(path.relative_to(REPO)),parent_config=str(PARENT.relative_to(REPO)),parent_run="TASK-011A Shared+Progress trial-28",**meta)
        item["config_sha256"]=digest(path); output.append(item)
    MANIFEST.write_text(json.dumps({"schema":"task011b-bundle-v1","production_count":52,"test_access":False,"rows":output},indent=2,sort_keys=True)+"\n")
    print(f"PREPARE PASS rows=52 manifest={MANIFEST.relative_to(REPO)}")

def manifest() -> dict[str,Any]:
    if not MANIFEST.is_file(): stop("manifest missing")
    m=json.loads(MANIFEST.read_text())
    if m.get("production_count") != 52 or len(m.get("rows",[])) != 52: stop("manifest is not 52 rows")
    return m

def validate() -> None:
    m,parent=manifest(),load(PARENT); expected=rows()
    if [(x["group"],x["variant"],x["seed"]) for x in m["rows"]] != [(x["group"],x["variant"],x["seed"]) for x in expected]: stop("manifest order mismatch")
    env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":"src"}
    for row in m["rows"]:
        path=REPO/row["config_path"]
        if not path.is_file() or digest(path) != row["config_sha256"]: stop(f"config SHA mismatch: {path}")
        c=load(path)
        if c["seed"] != row["seed"] or c["experiment_info"]["params"]["run_name"] != row["run_name"]: stop(f"config identity mismatch: {path}")
        if c["data"]["name"] != "wsm_ramps_semantic_dev_only_datamodule": stop(f"not DEV-only: {path}")
        p=subprocess.run([str(REPO/".venv/bin/chimera-ml"),"validate-config","--config-path",str(path)],cwd=REPO,env=env,capture_output=True,text=True)
        if p.returncode: stop(f"validate-config failed for {path}\n{p.stdout}\n{p.stderr}")
    for p in (PSEUDO,UNIFORM,NO_SEM,SHUFFLED):
        got=sha_cache_contract(p)
        if got["missing_counts"] != [2665,3660]: stop(f"missing counts mismatch: {p}")
        if p == NO_SEM and got["accepted_counts"] != [0,1801]: stop(f"no-sem counts mismatch: {got}")
        if p != NO_SEM and got["accepted_counts"] != [376,1801]: stop(f"accepted counts mismatch: {p} {got}")
    if digest(PSEUDO) != PSEUDO_SHA: stop("canonical pseudo cache SHA mismatch")
    sys.path.insert(0,str(REPO/"src"))
    import chimera_plugin
    import torch
    from fusion.data.wsm_ramps_semantic_datamodule import WSMRampsSemanticDataModule,WSMRampsSemanticDevOnlyDataModule
    common=deepcopy(parent["data"]["params"]); base=WSMRampsSemanticDataModule(**common); dev=WSMRampsSemanticDevOnlyDataModule(**common)
    bt=[x["meta"]["segment_id"] for x in base.train_dataset.base._samples]; dt=[x["meta"]["segment_id"] for x in dev.train_dataset.base._samples]
    bv=[x["meta"]["segment_id"] for x in base.val_dataset.base._samples]; dv=[x["meta"]["segment_id"] for x in dev.val_dataset.base._samples]
    if bt != dt or bv != dv or len(bt) != 6325 or len(bv) != 933: stop("TRAIN/DEV membership mismatch")
    if set(dev.val_dataloader()) != {"dev"}: stop("DEV-only dataloader exposes non-DEV")
    from fusion.models.av_r3_disease_query import WSMAVR3DiseaseQueryModel
    from fusion.loss.gradient_mtl_loss import WSMGradientMTLLoss
    from common.callbacks.wsm_gradient_mtl_callback import WSMGradientMTLCallback
    batch=next(iter(dev.train_dataloader()))
    for aware in (False,True):
        model_params=deepcopy(parent["model"]["params"]); model_params["task_aware_fusion"]=aware
        model=WSMAVR3DiseaseQueryModel(**model_params)
        if sum(p.numel() for p in model.parameters() if p.requires_grad) != 552775: stop("param count mismatch")
        for method in ("pcgrad","cagrad","gradnorm","dbmtl"):
            loss=WSMGradientMTLLoss(method=method,pseudo_scale=0.0,pcgrad_seed=42); loss.bind_model(model)
            wanted={"shared":14,"depression":18,"parkinson":18,"row":1} if aware else {"shared":23,"depression":14,"parkinson":14,"row":0}
            if loss.ownership_counts != wanted: stop(f"ownership mismatch {loss.ownership_counts}")
            model.zero_grad(set_to_none=True); value=loss(model(batch),batch); value.backward()
            if not torch.isfinite(value) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad): stop(f"nonfinite TRAIN smoke {method} {aware}")
        WSMGradientMTLCallback().on_fit_start(type("Trainer",(),{"model":model,"loss_fn":loss})())
    print("VALIDATE PASS rows=52 test_loader_accessed=False train_dev_equivalent=True")

def dry_run() -> None:
    m=manifest()
    for r in m["rows"]: print(f"{r['index']:02d} {r['group']} {r['variant']} seed={r['seed']} selector={r['selector']}")
    print("DRY-RUN PASS rows=52")

def run(resume: bool) -> None:
    m=manifest(); LEDGER.parent.mkdir(parents=True,exist_ok=True)
    ledger=json.loads(LEDGER.read_text()) if LEDGER.is_file() else {"schema":"task011b-execution-v1","rows":[]}; current={x["index"]:x for x in ledger["rows"]}
    env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":"src"}
    for r in m["rows"]:
        old=current.get(r["index"])
        if old and old.get("status")=="FAILED": stop(f"failed row {r['index']}; no automatic retry")
        if old and old.get("status")=="COMPLETE":
            if old.get("config_sha256") != r["config_sha256"]: stop(f"completed hash mismatch row {r['index']}")
            if not resume: stop(f"row {r['index']} complete; use --resume")
            continue
        if old and old.get("status")=="RUNNING": stop(f"interrupted row {r['index']}; no retry")
        entry={"index":r["index"],"group":r["group"],"variant":r["variant"],"seed":r["seed"],"run_name":r["run_name"],"config_path":r["config_path"],"config_sha256":r["config_sha256"],"status":"RUNNING"}
        current[r["index"]]=entry; LEDGER.write_text(json.dumps({"schema":"task011b-execution-v1","rows":[current[k] for k in sorted(current)]},indent=2)+"\n")
        log=LEDGER.parent/f"{r['index']:02d}_{r['variant']}_seed{r['seed']}.log"
        with log.open("w") as handle:
            p=subprocess.run([str(REPO/".venv/bin/chimera-ml"),"train","--config-path",str(REPO/r["config_path"])],cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT)
        entry.update({"status":"COMPLETE" if p.returncode==0 else "FAILED","returncode":p.returncode,"log_path":str(log.relative_to(REPO))})
        LEDGER.write_text(json.dumps({"schema":"task011b-execution-v1","rows":[current[k] for k in sorted(current)]},indent=2)+"\n")
        print(f"{entry['status']} {r['index']}/52 {r['variant']} seed={r['seed']}")
        if p.returncode: stop(f"production failed row {r['index']} returncode={p.returncode}; no retry")
    if len([x for x in current.values() if x.get("status")=="COMPLETE"]) != 52: stop("not 52 complete")
    print("RUN PASS rows=52")

DEV_KEYS = {
    "dev/depression/uar", "dev/depression/mf1", "dev/depression/score",
    "dev/parkinson/uar", "dev/parkinson/mf1", "dev/parkinson/score",
    "dev/mean_score",
}

def _dev_series(db: sqlite3.Connection, run_id: str) -> dict[str, list[tuple[int,float]]]:
    keys=[x[0] for x in db.execute(
        "SELECT DISTINCT key FROM metrics WHERE run_uuid=? AND key IN (?,?,?,?,?,?,?)",
        (run_id,*sorted(DEV_KEYS)))]
    if not keys:
        stop(f"no allowlisted DEV metrics for {run_id}")
    values={}
    for key in keys:
        values[key]=[(int(step),float(value)) for _,value,step in db.execute(
            "SELECT key,value,step FROM metrics WHERE run_uuid=? AND key=? ORDER BY step",
            (run_id,key))]
    return values

def _selected_dev(values: dict[str,list[tuple[int,float]]], selector_name: str) -> tuple[int, tuple[tuple[str,float],...]]:
    if selector_name not in values:
        stop(f"selector {selector_name} is missing from DEV metrics")
    epoch,_=max(values[selector_name],key=lambda x:x[1])
    selected=tuple(sorted((key,value) for key,entries in values.items()
                          for step,value in entries if step==epoch))
    return epoch,selected

def _artifact_signature(db: sqlite3.Connection, row: dict[str,Any],
                        run_id: str, actual_name: str, status: str) -> tuple[Any,...]:
    if status != "FINISHED":
        stop(f"non-FINISHED candidate {run_id}: {status}")
    root=REPO/"logs/wsm_mm_pd_dep_v1"/actual_name
    if not root.is_dir():
        stop(f"run artifact directory missing for {run_id}: {root}")
    snapshots=list(root.glob(Path(row["config_path"]).name))
    if len(snapshots) != 1 or digest(snapshots[0]) != row["config_sha256"]:
        stop(f"config identity mismatch for {run_id}: {snapshots}")
    config=load(snapshots[0])
    if config.get("seed") != row["seed"]:
        stop(f"seed mismatch for {run_id}")
    actual_mode = "task_aware" if config["model"]["params"].get("task_aware_fusion",True) else "shared"
    if actual_mode != row["model_mode"]:
        stop(f"model mode mismatch for {run_id}: {actual_mode} != {row['model_mode']}")
    if config["loss"]["params"].get("mode") != row["loss_mode"] or config["loss"]["params"].get("method") != row["loss_method"]:
        stop(f"loss identity mismatch for {run_id}")
    params=dict(db.execute("SELECT key,value FROM params WHERE run_uuid=?",(run_id,)).fetchall())
    if abs(float(params.get("optimizer_lr/group0","nan"))-float(config["optimizer"]["params"]["lr"])) > 1e-15:
        stop(f"optimizer lr mismatch for {run_id}")
    if abs(float(params.get("optimizer_weight_decay/group0","nan"))-float(config["optimizer"]["params"]["weight_decay"])) > 1e-15:
        stop(f"optimizer weight decay mismatch for {run_id}")
    epoch,selected=_selected_dev(_dev_series(db,run_id),row["selector"])
    candidates=list((root/"checkpoints").glob(f"epoch={epoch}_*.pt"))
    if len(candidates) != 1:
        stop(f"selected checkpoint identity is not unique for {run_id}: {candidates}")
    checkpoint_path=candidates[0]
    return (status,row["group"],row["variant"],row["seed"],row["run_name"],
            row["config_sha256"],actual_mode,row["loss_mode"],row["loss_method"],
            float(config["optimizer"]["params"]["lr"]),
            float(config["optimizer"]["params"]["weight_decay"]),epoch,
            checkpoint_path.name,digest(checkpoint_path),selected)

def resolve_runs(manifest_rows: list[dict[str, Any]]) -> dict[int,dict[str,Any]]:
    db=sqlite3.connect(f"file:{REPO / 'logs/mlflow.db'}?mode=ro",uri=True)
    tags=db.execute("SELECT run_uuid,value FROM tags WHERE key='mlflow.runName'").fetchall()
    statuses=dict(db.execute("SELECT run_uuid,status FROM runs").fetchall())
    resolved={}
    referenced=set()
    for row in manifest_rows:
        prefix=row["run_name"]
        pattern=re.compile(r"^"+re.escape(prefix)+r"(?:_|$)")
        candidates=[(run_id,name,statuses.get(run_id,"")) for run_id,name in tags
                    if pattern.match(name) and statuses.get(run_id)=="FINISHED"]
        if not candidates:
            db.close()
            stop(f"no FINISHED candidate for row {row['index']} {prefix}")
        checked=[(run_id,name,status,_artifact_signature(db,row,run_id,name,status))
                 for run_id,name,status in candidates]
        signatures={item[3] for item in checked}
        if len(signatures) != 1:
            db.close()
            stop(f"alias equivalence mismatch for row {row['index']}: {[(x[0],x[3]) for x in checked]}")
        ids={x[0] for x in checked}
        if referenced.intersection(ids):
            db.close()
            stop(f"MLflow alias reused across manifest rows: {row['index']}")
        referenced.update(ids)
        canonical=min(ids)
        selected=next(x for x in checked if x[0]==canonical)
        resolved[row["index"]]={"canonical_run_id":canonical,"canonical_name":selected[1],
                                "alias_run_ids":sorted(ids-{canonical}),
                                "run_status":selected[2],"signature":selected[3]}
    db.close()
    if len(resolved) != 52 or len({x["signature"] for x in resolved.values()}) != 52:
        stop(f"scientific result count failure: rows={len(resolved)} results={len({x['signature'] for x in resolved.values()})}")
    return resolved

def audit_collection(manifest_rows: list[dict[str, Any]]) -> dict[int,dict[str,Any]]:
    resolved=resolve_runs(manifest_rows)
    for row in manifest_rows:
        item=resolved[row["index"]]
        print(f"mapping {row['index']:02d} {row['run_name']} -> {item['canonical_run_id']} aliases={item['alias_run_ids']} {item['run_status']}")
    alias_rows=sum(bool(x["alias_run_ids"]) for x in resolved.values())
    referenced=sum(1+len(x["alias_run_ids"]) for x in resolved.values())
    print(f"COLLECTION AUDIT PASS rows=52 scientific_results=52 referenced_mlflow_runs={referenced} duplicate_alias_rows={alias_rows} test_keys_requested=False")
    return resolved

def mlflow_dev(db: sqlite3.Connection, run_id: str, run_name: str, selector_name: str) -> dict[str,float]:
    values=_dev_series(db,run_id)
    epoch,_=_selected_dev(values,selector_name)
    selected={key:value for key,entries in values.items() for step,value in entries if step==epoch}
    selected["selected_epoch"]=float(epoch)
    return selected

def checkpoint(actual_name: str, epoch: int) -> tuple[str|None,str|None,int|None]:
    root=REPO/"logs/wsm_mm_pd_dep_v1"/actual_name
    found=list((root/"checkpoints").glob(f"epoch={epoch}_*.pt"))
    if len(found) != 1:
        stop(f"selected checkpoint is not unique for {actual_name}: {found}")
    p=found[0]
    import torch
    state=torch.load(p,map_location="cpu",weights_only=False)
    vals=state.get("model_state_dict",{})
    return str(p.relative_to(REPO)),digest(p),sum(v.numel() for v in vals.values() if hasattr(v,"numel"))

def write_report(result: list[dict[str,Any]]) -> None:
    grouped: dict[str,list[dict[str,Any]]] = {}
    for row in result:
        grouped.setdefault(row["variant"], []).append(row)
    for value in grouped.values():
        value.sort(key=lambda row: row["seed"])

    frozen_parent = {
        42: {"mean": 0.824465, "d_score": 0.765324, "p_score": 0.883606},
        43: {"mean": 0.771324, "d_score": 0.724496, "p_score": 0.818152},
        44: {"mean": 0.784910, "d_score": 0.731448, "p_score": 0.838372},
        45: {"mean": 0.787202, "d_score": 0.739733, "p_score": 0.834672},
        46: {"mean": 0.786671, "d_score": 0.727451, "p_score": 0.845891},
    }
    frozen_seed42 = {
        "mean": 0.8336994328,
        "epoch": 7,
        "checkpoint_sha256": "ba02ea341800d8d6330ecca32e8a31dde3afaf61e5fb1e1eca4fd144d1ff0346",
        "d_score": 0.749256,
        "p_score": 0.918143,
    }
    parent_rows = {row["seed"]: row for row in grouped["shared_progress_parent"]}
    tuned_parent = {42: frozen_seed42}
    for seed, row in parent_rows.items():
        metrics = row["metrics"]
        tuned_parent[seed] = {
            "mean": float(metrics["dev/mean_score"]),
            "d_score": float(metrics["dev/depression/score"]),
            "p_score": float(metrics["dev/parkinson/score"]),
        }

    def f(value: float) -> str:
        return f"{value:.6f}"

    def paired_stats(values: list[float]) -> tuple[float,float,float,float,int]:
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        half = T4 * sd / math.sqrt(len(values))
        return mean, sd, mean - half, mean + half, sum(value > 0 for value in values)

    def task_triplet(row: dict[str,Any], task: str) -> str:
        prefix = f"dev/{task}/"
        return "/".join(f(row["metrics"][prefix + key]) for key in ("uar", "mf1", "score"))

    alias_rows = [row for row in result if row.get("alias_run_ids")]
    referenced_ids = sum(1 + len(row.get("alias_run_ids", [])) for row in result)
    lines = [
        "# POST-CLOSURE DEV-ONLY FULL ABLATION/MTL SUITE — NOT PART OF THE FROZEN STAGE-7/FINAL-TEST EVIDENCE",
        "",
        "This is descriptive DEV-only evidence around TASK-011A Shared+Progress trial-28. It does not revise Stage-7, Final-Test, or paper roles.",
        "",
        "## POST-PRODUCTION REPORTING-ONLY COLLECTION RECOVERY",
        "",
        f"Collection integrity: manifest rows {len(result)}/52; scientific production results {len(result)}/52; referenced MLflow run IDs {referenced_ids}; duplicate-alias rows {len(alias_rows)}.",
        "The accepted resolver uses anchored run-name prefixes, FINISHED status, frozen config/artifact identity, and the lexicographically smallest ID for an equivalent alias set.",
    ]
    if len(alias_rows) == 1:
        alias = alias_rows[0]
        alias_id = alias["alias_run_ids"][0]
        metrics = alias["metrics"]
        lines += [
            f"Row 34 aliases {alias['run_id']} and {alias_id} were equivalent FINISHED aliases for one frozen result; canonical ID is {alias['run_id']} and alias ID is {alias_id}.",
            f"Equivalence included config SHA 660cac8293e60e28648c4225edf78486dbb06087a9459d730bb49ada157d5be8, selected epoch {alias['selected_epoch']}, checkpoint SHA {alias['checkpoint_sha256']}, and identical DEV-only selected metrics D {f(metrics['dev/depression/uar'])}/{f(metrics['dev/depression/mf1'])}/{f(metrics['dev/depression/score'])}, P {f(metrics['dev/parkinson/uar'])}/{f(metrics['dev/parkinson/mf1'])}/{f(metrics['dev/parkinson/score'])}, Mean {f(metrics['dev/mean_score'])}.",
            "Canonicalization is bookkeeping only and does not count the alias as another production run.",
        ]

    lines += ["", "## 1. Five-seed tuned parent confirmation", "", "Only allowlisted DEV metric keys were collected."]
    lines.append(f"seed42 frozen tuned parent: D UAR/F1/Score unavailable/unavailable/{f(frozen_seed42['d_score'])}; P UAR/F1/Score unavailable/unavailable/{f(frozen_seed42['p_score'])}; Mean {f(frozen_seed42['mean'])}; epoch {frozen_seed42['epoch']}; checkpoint SHA {frozen_seed42['checkpoint_sha256']}.")
    for seed in (43, 44, 45, 46):
        row = parent_rows[seed]
        lines.append(f"seed{seed}: D {task_triplet(row, 'depression')}; P {task_triplet(row, 'parkinson')}; Mean {f(row['metrics']['dev/mean_score'])}; epoch {row['selected_epoch']}; checkpoint SHA {row['checkpoint_sha256']}.")
    tuned_means = [tuned_parent[seed]["mean"] for seed in (42,43,44,45,46)]
    parent_mean, parent_sd, parent_lo, parent_hi, _ = paired_stats(tuned_means)
    lines.append(f"Tuned parent five-seed Mean mean/std/min/max/range/95% t CI(df=4): {f(parent_mean)}/{f(parent_sd)}/{f(min(tuned_means))}/{f(max(tuned_means))}/{f(max(tuned_means)-min(tuned_means))}/[{f(parent_lo)},{f(parent_hi)}].")
    lines.append("Frozen Shared trial012 five-seed Mean mean/std/95% t CI(df=4) from committed Stage-7 DEV evidence: 0.790914/0.019857/[0.766258,0.815571].")

    mean_deltas = [tuned_parent[seed]["mean"] - frozen_parent[seed]["mean"] for seed in (42,43,44,45,46)]
    d_deltas = [tuned_parent[seed]["d_score"] - frozen_parent[seed]["d_score"] for seed in (42,43,44,45,46)]
    p_deltas = [tuned_parent[seed]["p_score"] - frozen_parent[seed]["p_score"] for seed in (42,43,44,45,46)]
    mean_delta, mean_delta_sd, mean_lo, mean_hi, mean_wins = paired_stats(mean_deltas)
    d_delta, _, _, _, d_wins = paired_stats(d_deltas)
    p_delta, _, _, _, p_wins = paired_stats(p_deltas)
    lines += [
        "Tuned-minus-frozen Shared Mean deltas by seed (42/43/44/45/46): " + ", ".join(f"{value:+.10f}" for value in mean_deltas) + ".",
        f"Paired Mean delta mean/sample SD/95% exploratory t CI(df=4)/wins: {mean_delta:+.10f}/{mean_delta_sd:.10f}/[{mean_lo:+.10f},{mean_hi:+.10f}]/{mean_wins}/5.",
        f"Parent task-score comparison versus frozen Shared: D Score mean delta {d_delta:+.10f}, wins {d_wins}/5; P Score mean delta {p_delta:+.10f}, wins {p_wins}/5.",
        "This is descriptive post-closure DEV evidence, not a Final-Test superiority claim.",
    ]

    parent_by_seed = {seed: tuned_parent[seed]["mean"] for seed in tuned_parent}

    def two_task_rows(variants: list[str]) -> None:
        for variant in variants:
            rows_for_variant = grouped[variant]
            d_uar = statistics.mean(row["metrics"]["dev/depression/uar"] for row in rows_for_variant)
            d_f1 = statistics.mean(row["metrics"]["dev/depression/mf1"] for row in rows_for_variant)
            d_score = statistics.mean(row["metrics"]["dev/depression/score"] for row in rows_for_variant)
            p_uar = statistics.mean(row["metrics"]["dev/parkinson/uar"] for row in rows_for_variant)
            p_f1 = statistics.mean(row["metrics"]["dev/parkinson/mf1"] for row in rows_for_variant)
            p_score = statistics.mean(row["metrics"]["dev/parkinson/score"] for row in rows_for_variant)
            means = [row["metrics"]["dev/mean_score"] for row in rows_for_variant]
            deltas = [means[i] - parent_by_seed[row["seed"]] for i, row in enumerate(rows_for_variant)]
            lines.append(
                f"{variant}: D UAR/F1/Score mean {f(d_uar)}/{f(d_f1)}/{f(d_score)}; P UAR/F1/Score mean {f(p_uar)}/{f(p_f1)}/{f(p_score)}; Mean mean/sample SD {f(statistics.mean(means))}/{f(statistics.stdev(means))}; paired Mean deltas (42/43/44) "
                + ", ".join(f"{value:+.6f}" for value in deltas)
                + f"; wins {sum(value > 0 for value in deltas)}/3."
            )

    lines += ["", "## 2. Core component ablations", "", "Each row reports three-seed DEV task-level aggregates and paired Mean deltas against the same-seed tuned Shared+Progress parent."]
    two_task_rows(["no_direct_pseudo", "uniform_reliability", "no_semantic_depression", "no_video_online", "no_audio_online"])

    lines += ["", "## 3. Architecture and shuffled-pseudo controls", ""]
    two_task_rows(["task_aware_r4_progress", "shuffled_pseudo"])

    lines += ["", "## 4. Single-task vs joint-MTL controls", "", "Single-task controls report only their selected task metrics; joint Mean is not reported or used for these comparisons."]
    for variant, task, label in (("depression_only", "depression", "D"), ("parkinson_only", "parkinson", "P")):
        rows_for_variant = grouped[variant]
        key = f"dev/{task}/score"
        scores = [row["metrics"][key] for row in rows_for_variant]
        deltas = [scores[i] - tuned_parent[row["seed"]]["d_score" if task == "depression" else "p_score"] for i, row in enumerate(rows_for_variant)]
        mean, sd, _, _, _ = paired_stats(scores)
        wins = sum(value > 0 for value in deltas)
        lines.append(f"{variant}: selected {label} UAR/F1/Score by seed (42/43/44): " + ", ".join(task_triplet(row, task) for row in rows_for_variant) + f"; {label} Score mean/sample SD {f(mean)}/{f(sd)}; paired {label}-only {label} Score minus joint-parent {label} Score deltas " + ", ".join(f"{value:+.6f}" for value in deltas) + f"; wins {wins}/3; joint Mean —.")

    lines += ["", "## 5. Scalarization / balancing substitutions", ""]
    two_task_rows(["equal", "static_stch", "ra_stch"])

    lines += ["", "## 6. Gradient-MTL substitutions", "", "Common hyperparameters were tuned under Progress. These are frozen-recipe substitution audits, not equal-budget globally tuned rankings of balancing/gradient-MTL methods.", ""]
    two_task_rows(["gradnorm", "pcgrad", "cagrad", "dbmtl"])

    lines += ["", "## 7. Scope / limitations / Test quarantine", "", "- All reported metrics and comparisons are DEV-only; checkpoint selection remains by each run's configured DEV selector.", "- No Test loader, Test metric, mixed raw summary, or Final-Test artifact was read by this collection or synthesis.", "- D-only/P-only are not deployable two-task models; their joint Mean is deliberately absent and no fake Mean comparison is made.", "- Common Progress-tuned hyperparameters make Groups F/G frozen-recipe audits, not globally tuned rankings.", "- Online modality removals retain historical offline pseudo evidence.", "- All three-seed statistics and any paired t intervals are descriptive/exploratory; no multiplicity-corrected significance or universal usefulness/uselessness claim is authorized.", "- No final-method promotion, demotion, or paper-role revision is authorized."]
    REPORT.write_text("\n".join(lines) + "\n")

def collect() -> None:
    m=manifest()
    ledger=json.loads(LEDGER.read_text()) if LEDGER.is_file() else {}
    if len([x for x in ledger.get("rows",[]) if x.get("status")=="COMPLETE"]) != 52:
        stop("collect requires 52 COMPLETE rows")
    resolved=audit_collection(m["rows"])
    db=sqlite3.connect(f"file:{REPO / 'logs/mlflow.db'}?mode=ro",uri=True)
    out=[]
    for r in m["rows"]:
        item=resolved[r["index"]]
        run_id=item["canonical_run_id"]
        actual=item["canonical_name"]
        metrics=mlflow_dev(db,run_id,actual,r["selector"])
        epoch=int(metrics.pop("selected_epoch"))
        path,csha,params=checkpoint(actual,epoch)
        out.append({k:r[k] for k in ("index","group","variant","seed","run_name","selector","config_path","config_sha256")}|{
            "run_id":run_id,"alias_run_ids":item["alias_run_ids"],"mlflow_run_name":actual,
            "run_status":item["run_status"],"selected_epoch":epoch,"metrics":metrics,
            "checkpoint_path":path,"checkpoint_sha256":csha,"parameters":params})
    db.close()
    RESULTS.parent.mkdir(parents=True,exist_ok=True)
    RESULTS.write_text(json.dumps({"schema":"task011b-dev-only-results-v1","test_access":False,
        "manifest_rows":52,"scientific_results":52,
        "referenced_mlflow_runs":sum(1+len(x["alias_run_ids"]) for x in resolved.values()),
        "duplicate_alias_rows":sum(bool(x["alias_run_ids"]) for x in resolved.values()),
        "rows":out},indent=2,sort_keys=True)+"\n")
    write_report(out)
    print(f"COLLECT PASS rows=52 scientific_results=52 test_metrics_read=False report={REPORT.relative_to(REPO)}")

def main() -> None:
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True)
    for x in ("prepare","validate","dry-run","run","resume","audit-collection","collect"): g.add_argument("--"+x,action="store_true")
    a=p.parse_args()
    if a.prepare: prepare()
    elif a.validate: validate()
    elif a.dry_run: dry_run()
    elif a.run: run(False)
    elif a.resume: run(True)
    elif a.audit_collection: audit_collection(manifest()["rows"])
    else: collect()
if __name__=="__main__": main()
