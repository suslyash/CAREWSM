#!/usr/bin/env python3
"""Failure-tolerant sequential worker for audio_turns_combined."""

from __future__ import annotations
import csv
import fcntl
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path("logs/audio_turns_combined")
CONFIG_ROOT = Path("configs/audio_turns_combined")
LEDGER = ROOT / "reports/run_ledger.csv"
STATUS = ROOT / "launcher/status.json"
LOCK = ROOT / "launcher/worker.lock"
ATTEMPTS = ROOT / "reports/run_attempts.csv"
PYTHON = Path(".venv/bin/python")
CHIMERA = Path(".venv/bin/chimera-ml")
FIELDS = (
    "stage",
    "experiment_id",
    "feature_combination",
    "feature_families",
    "representation",
    "model",
    "hidden_dim",
    "fold",
    "seed",
    "status",
    "attempt_number",
    "start_time",
    "end_time",
    "exit_code",
    "failure_reason",
    "runtime_directory",
    "checkpoint_path",
    "selected_epoch",
    "selected_val_uar",
    "mlflow_run_id",
    "source_commit",
)
LETTERS = {
    "emotionmsp_embeds": "A",
    "emotionmsp_trajectories": "B",
    "trillsson4_1": "C",
    "wav2vec_xlsr300m": "D",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def key(row):
    return row["experiment_id"], row["fold"]


def load_rows():
    if not LEDGER.exists():
        return []
    with LEDGER.open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def save_rows(rows):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    temporary = LEDGER.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(LEDGER)


def set_status(**values):
    data = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    data.update(values)
    if values.get("state") in {"STARTING", "RUNNING"} and "error" not in values:
        data.pop("error", None)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def counts(rows):
    return {
        "total": len(rows),
        "completed": sum(r["status"] == "COMPLETED" for r in rows),
        "failed": sum(r["status"] == "FAILED" for r in rows),
        "pending": sum(r["status"] == "PENDING" for r in rows),
        "blocked": sum(r["status"] == "BLOCKED" for r in rows),
        "running": sum(r["status"] == "RUNNING" for r in rows),
    }


def update_progress(rows, **values):
    set_status(**counts(rows), **values)


def append_attempt(row, log_path):
    ATTEMPTS.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if ATTEMPTS.exists():
        with ATTEMPTS.open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
    item = {field: row.get(field, "") for field in FIELDS}
    item["log_path"] = str(log_path)
    history.append(item)
    temporary = ATTEMPTS.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(item))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(ATTEMPTS)


def failure_reason(log_path):
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        if any(token in line for token in ("Error:", "Exception:", "failed")):
            return line.strip()[-500:]
    return "training exit or artifact verification failed"


def identity(path):
    data = yaml.safe_load(path.read_text())
    params = data["data"]["params"]
    info = data["experiment_info"]["params"]
    configured_families = params["feature_family"]
    families = [configured_families] if isinstance(configured_families, str) else configured_families
    return {
        "stage": path.parent.name,
        "experiment_id": info["run_name"].rsplit("_fold", 1)[0],
        "feature_combination": "".join(LETTERS[x] for x in families),
        "feature_families": "+".join(families),
        "representation": params["pooling"],
        "model": data["model"]["name"],
        "hidden_dim": str(data["model"]["params"]["hidden_dim"]),
        "fold": str(params["fold"]),
        "seed": str(data["seed"]),
        "run_name": info["run_name"],
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def runtime(name):
    found = sorted((ROOT / "audio_turns_combined").glob(name + "_*"))
    return found[0] if len(found) == 1 else None


def verified(path):
    data = yaml.safe_load(path.read_text())
    name = data["experiment_info"]["params"]["run_name"]
    directory = runtime(name)
    if (
        directory is None
        or not (directory / "summary.txt").exists()
        or not (directory / "train.log").exists()
    ):
        return False
    return (
        "Training complete" in (directory / "train.log").read_text()
        and len(
            [p for p in (directory / "checkpoints").glob("*.pt") if p.name != "last.pt"]
        )
        == 1
    )


def run_one(path, rows):
    ident = identity(path)
    old = next(
        (r for r in rows if key(r) == (ident["experiment_id"], ident["fold"])), None
    )
    if old and old["status"] == "COMPLETED" and verified(path):
        return
    attempt = int(old["attempt_number"] or 0) + 1 if old else 1
    row = {field: "" for field in FIELDS}
    row.update(
        {field: value for field, value in ident.items() if field != "run_name"},
        status="RUNNING",
        attempt_number=str(attempt),
        start_time=now(),
    )
    rows[:] = [r for r in rows if key(r) != key(row)] + [row]
    save_rows(rows)
    update_progress(
        rows, state="RUNNING", current_stage=ident["stage"],
        current_experiment=ident["experiment_id"], current_fold=ident["fold"],
    )
    log = (
        ROOT
        / "launcher"
        / (ident["experiment_id"] + f"_fold{ident['fold']}_attempt{attempt}.log")
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        process = subprocess.run(
            [str(CHIMERA), "train", "-c", str(path)],
            stdout=output,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            },
        )
    row.update(
        end_time=now(),
        exit_code=str(process.returncode),
        runtime_directory=str(runtime(ident["run_name"]) or ""),
    )
    if process.returncode == 0 and verified(path):
        row["status"] = "COMPLETED"
    else:
        row.update(status="FAILED", failure_reason=failure_reason(log))
    rows[:] = [r for r in rows if key(r) != key(row)] + [row]
    save_rows(rows)
    append_attempt(row, log)
    subprocess.run(
        [str(PYTHON), "scripts/audio_turns_combined/collect_results.py"], check=False
    )
    collected = ROOT / "reports/all_results_by_fold.csv"
    if row["status"] == "COMPLETED" and collected.exists():
        with collected.open(newline="", encoding="utf-8") as handle:
            evidence = next(
                (
                    item
                    for item in csv.DictReader(handle)
                    if item["experiment_id"] == row["experiment_id"]
                    and item["fold"] == row["fold"]
                ),
                None,
            )
        if evidence is not None:
            row.update(
                {
                    "checkpoint_path": evidence["checkpoint_path"],
                    "selected_epoch": evidence["selected_epoch"],
                    "selected_val_uar": evidence["selected_val_uar"],
                    "mlflow_run_id": evidence["mlflow_run_id"],
                }
            )
            save_rows(rows)


def sync_paths(paths, rows):
    changed = False
    for path in paths:
        ident = identity(path)
        existing = next(
            (row for row in rows if key(row) == (ident["experiment_id"], ident["fold"])),
            None,
        )
        if existing is not None:
            if existing["status"] == "PENDING":
                for field, value in ident.items():
                    if field != "run_name":
                        existing[field] = value
                changed = True
            continue
        row = {field: "" for field in FIELDS}
        row.update(
            {field: value for field, value in ident.items() if field != "run_name"},
            status="PENDING",
            attempt_number="0",
        )
        rows.append(row)
        changed = True
    if changed:
        save_rows(rows)


def stage_done(stage, rows):
    paths = sorted((CONFIG_ROOT / stage).glob("*.yaml"))
    return bool(paths) and all(
        any(
            r["experiment_id"] == p.stem.rsplit("_fold", 1)[0]
            and r["fold"] == p.stem.rsplit("_fold", 1)[1]
            and r["status"] == "COMPLETED"
            for r in rows
        )
        for p in paths
    )


def scores(rows, stages):
    result = []
    for experiment in sorted(
        {
            r["experiment_id"]
            for r in rows
            if r["stage"] in stages and r["status"] == "COMPLETED"
        }
    ):
        group = [
            r
            for r in rows
            if r["experiment_id"] == experiment and r["status"] == "COMPLETED"
        ]
        if len(group) == 5:
            result.append(
                (
                    sum(float(r.get("selected_val_uar") or 0) for r in group) / 5,
                    experiment,
                    group[0],
                )
            )
    return sorted(result, reverse=True)


def generate(stage, entries):
    if entries:
        subprocess.run(
            [
                str(PYTHON),
                "scripts/audio_turns_combined/generate_configs.py",
                stage,
                "--entries",
                *entries,
            ],
            check=True,
        )


def prepare(stage, rows):
    if stage == "feature_screening":
        subprocess.run(
            [str(PYTHON), "scripts/audio_turns_combined/generate_configs.py", "stage1"],
            check=True,
        )
        return
    if stage == "gated_fusion" and stage_done("feature_screening", rows):
        codes = []
        for _, exp, _ in scores(rows, {"feature_screening"}):
            code = exp.removeprefix("screen_").removesuffix("_mean_mlp32")
            if len(code) > 1 and code not in codes:
                codes.append(code)
        if "AC" not in codes:
            codes.append("AC")
        generate("stage2", [x + ":mean" for x in codes[:4]])
    elif stage == "mean_std" and stage_done("gated_fusion", rows):
        choices = []
        for _, _, r in scores(rows, {"feature_screening", "gated_fusion"}):
            item = f"{r['feature_combination']}:{r['representation']}:{r['model']}:{r['hidden_dim']}"
            if r["feature_combination"] not in [x.split(":")[0] for x in choices]:
                choices.append(item)
        if "AC" not in [x.split(":")[0] for x in choices]:
            choices.append("AC:mean:carewsm_static_mlp:32")
        generate("stage3", choices[:4])
    elif stage == "model_tuning" and stage_done("mean_std", rows):
        ranked = scores(rows, {"feature_screening", "gated_fusion", "mean_std"})
        if ranked:
            r = ranked[0][2]
            generate(
                "stage4",
                [
                    f"{r['feature_combination']}:{r['representation']}:64:{r['model']}",
                    f"{r['feature_combination']}:{r['representation']}:64:carewsm_static_residual_mlp",
                ],
            )
    elif stage == "mtl_strategies" and stage_done("model_tuning", rows):
        ranked = scores(
            rows, {"feature_screening", "gated_fusion", "mean_std", "model_tuning"}
        )
        if ranked:
            r = ranked[0][2]
            generate(
                "stage5",
                [
                    f"{r['feature_combination']}:{r['representation']}:{r['hidden_dim']}:{r['model']}:{s}"
                    for s in ("equal", "stch", "pcgrad", "dbmtl")
                ],
            )


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = load_rows()
        update_progress(rows, state="STARTING")
        for stage in (
            "feature_screening",
            "gated_fusion",
            "mean_std",
            "model_tuning",
            "mtl_strategies",
        ):
            prepare(stage, rows)
            paths = list((CONFIG_ROOT / stage).glob("*.yaml"))
            sync_paths(paths, rows)
            if stage == "feature_screening":
                order = {
                    code: index
                    for index, code in enumerate(
                        (
                            "A",
                            "B",
                            "C",
                            "D",
                            "AB",
                            "AC",
                            "AD",
                            "BC",
                            "BD",
                            "CD",
                            "ABC",
                            "ABD",
                            "ACD",
                            "BCD",
                            "ABCD",
                        )
                    )
                }
                paths.sort(
                    key=lambda path: (
                        order[identity(path)["feature_combination"]],
                        int(identity(path)["fold"]),
                    )
                )
            else:
                paths.sort()
            for path in paths:
                run_one(path, rows)
                update_progress(rows, current_stage=stage, state="RUNNING")
            update_progress(rows, current_stage=stage, state="STAGE_COMPLETE")
        progress = counts(rows)
        terminal = progress["completed"] + progress["failed"] + progress["blocked"]
        state = (
            "COMPLETED"
            if progress["completed"] == progress["total"]
            else "COMPLETED_WITH_FAILURES"
            if terminal == progress["total"]
            else "INTERRUPTED"
        )
        update_progress(rows, state=state)
        subprocess.run(
            [str(PYTHON), "scripts/audio_turns_combined/collect_results.py"],
            check=False,
        )


if __name__ == "__main__":
    try:
        main()
    except BlockingIOError:
        set_status(
            state="RUNNING", error="another audio_turns_combined worker holds the lock"
        )
        raise
    except Exception as exc:
        set_status(state="INTERRUPTED", error=f"{type(exc).__name__}: {exc}")
        raise
