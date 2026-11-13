#!/usr/bin/env python3
"""Sequential, locked, resumable runner for the 25 production jobs."""

import argparse, csv, fcntl, json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path("logs/audio_interpretable_feature_fusion")
CONFIGS = Path("configs/audio_interpretable_feature_fusion")
LEDGER = ROOT / "reports/run_ledger.csv"
FIELDS = [
    "experiment_id",
    "combination",
    "fold",
    "status",
    "attempt",
    "start_time",
    "end_time",
    "exit_code",
    "source_commit",
    "config_path",
    "runtime",
    "error",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def load():
    return (
        list(csv.DictReader(LEDGER.open(newline="", encoding="utf-8")))
        if LEDGER.exists()
        else []
    )


def save(rows):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(LEDGER)


def identity(path):
    c = yaml.safe_load(path.read_text())
    p = c["experiment_info"]["params"]
    name = p["run_name"]
    return {
        "experiment_id": name.rsplit("_fold", 1)[0],
        "combination": name.split("_")[1],
        "fold": str(c["data"]["params"]["fold"]),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "config_path": str(path),
    }


def runtime(name):
    found = sorted((ROOT / "audio_interpretable_feature_fusion").glob(name + "_*/"))
    return found[-1] if found else None


def verified(path):
    name = yaml.safe_load(path.read_text())["experiment_info"]["params"]["run_name"]
    d = runtime(name)
    return bool(
        d
        and (d / "train.log").exists()
        and (d / "summary.txt").exists()
        and list((d / "checkpoints").glob("epoch=*.pt"))
    )


def run_one(path, rows):
    item = identity(path)
    old = next(
        (
            r
            for r in rows
            if r["experiment_id"] == item["experiment_id"] and r["fold"] == item["fold"]
        ),
        None,
    )
    if old and old["status"] == "COMPLETED" and verified(path):
        return
    attempt = int(old["attempt"] or 0) + 1 if old else 1
    row = {f: "" for f in FIELDS}
    row.update(item, status="RUNNING", attempt=str(attempt), start_time=now())
    rows[:] = [
        r
        for r in rows
        if not (
            r["experiment_id"] == item["experiment_id"] and r["fold"] == item["fold"]
        )
    ] + [row]
    save(rows)
    log = (
        ROOT
        / "launcher"
        / f"{item['experiment_id']}_fold{item['fold']}_attempt{attempt}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as h:
        result = subprocess.run(
            [".venv/bin/chimera-ml", "train", "-c", str(path)],
            stdout=h,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            },
        )
    row.update(
        status="COMPLETED" if result.returncode == 0 and verified(path) else "FAILED",
        end_time=now(),
        exit_code=str(result.returncode),
        runtime=str(runtime(item["experiment_id"] + "_fold" + item["fold"]) or ""),
    )
    if row["status"] == "FAILED":
        row["error"] = log.read_text(errors="replace")[-1200:]
    rows[:] = [
        r
        for r in rows
        if not (
            r["experiment_id"] == item["experiment_id"] and r["fold"] == item["fold"]
        )
    ] + [row]
    save(rows)


def prepare_directories():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "launcher").mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["preflight", "run", "status"])
    a = ap.parse_args()
    prepare_directories()
    paths = sorted(CONFIGS.glob("*.yaml"))
    if a.command == "preflight":
        print(
            json.dumps(
                {
                    "configs": len(paths),
                    "expected": 25,
                    "folds": sorted({identity(p)["fold"] for p in paths}),
                    "combinations": sorted({identity(p)["combination"] for p in paths}),
                },
                indent=2,
            )
        )
        return
    rows = load()
    if a.command == "status":
        print(
            json.dumps(
                {
                    "total": len(rows),
                    "completed": sum(r["status"] == "COMPLETED" for r in rows),
                    "failed": sum(r["status"] == "FAILED" for r in rows),
                    "running": sum(r["status"] == "RUNNING" for r in rows),
                },
                indent=2,
            )
        )
        return
    with (ROOT / "launcher/worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path in paths:
            run_one(path, rows)


if __name__ == "__main__":
    main()
