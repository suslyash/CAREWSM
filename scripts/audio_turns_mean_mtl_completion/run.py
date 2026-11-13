#!/usr/bin/env python3
"""Isolated, resumable runner for the 20 issue-109 jobs."""

import argparse
import csv
import fcntl
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path("logs/audio_turns_combined_mtl_completion")
CONFIGS = Path("configs/audio_turns_combined_mtl_completion")
FIELDS = ["experiment_id", "method", "fold", "status", "attempt", "start_time",
          "end_time", "exit_code", "source_commit", "config_path", "runtime", "error"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rows() -> list[dict[str, str]]:
    path = ROOT / "reports/run_ledger.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(items: list[dict[str, str]]) -> None:
    path = ROOT / "reports/run_ledger.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(items)


def identity(path: Path) -> dict[str, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = config["experiment_info"]["params"]["run_name"]
    return {"experiment_id": name.rsplit("_fold", 1)[0],
            "method": config["loss"]["params"]["method"],
            "fold": str(config["data"]["params"]["fold"]),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "config_path": str(path)}


def runtime(name: str) -> Path | None:
    found = sorted((ROOT / "audio_turns_combined_mtl_completion").glob(name + "_*/"))
    return found[-1] if found else None


def done(path: Path) -> bool:
    name = yaml.safe_load(path.read_text())["experiment_info"]["params"]["run_name"]
    directory = runtime(name)
    return bool(directory and (directory / "train.log").exists() and
                (directory / "summary.txt").exists() and
                list((directory / "checkpoints").glob("epoch=*.pt")))


def run_one(path: Path, ledger: list[dict[str, str]]) -> None:
    item = identity(path)
    current = next((row for row in ledger if row["experiment_id"] == item["experiment_id"]
                    and row["fold"] == item["fold"]), None)
    if current and current["status"] == "COMPLETED" and done(path):
        return
    attempt = int(current["attempt"] if current else 0) + 1
    row = {field: "" for field in FIELDS}
    row.update(item, status="RUNNING", attempt=str(attempt), start_time=now())
    ledger[:] = [r for r in ledger if not (r["experiment_id"] == item["experiment_id"] and r["fold"] == item["fold"])] + [row]
    save(ledger)
    log = ROOT / "launcher" / f"{item['experiment_id']}_fold{item['fold']}_attempt{attempt}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run([".venv/bin/chimera-ml", "train", "-c", str(path)],
                                stdout=handle, stderr=subprocess.STDOUT,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")})
    row.update(status="COMPLETED" if result.returncode == 0 and done(path) else "FAILED",
               end_time=now(), exit_code=str(result.returncode), runtime=str(runtime(item["experiment_id"] + "_fold" + item["fold"]) or ""))
    if row["status"] == "FAILED":
        row["error"] = log.read_text(errors="replace")[-1000:]
    ledger[:] = [r for r in ledger if not (r["experiment_id"] == item["experiment_id"] and r["fold"] == item["fold"])] + [row]
    save(ledger)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run", "status"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    paths = sorted(CONFIGS.glob("*.yaml"))
    if args.command == "preflight":
        print(json.dumps({"configs": len(paths), "expected": 20,
                          "methods": sorted({identity(p)["method"] for p in paths}),
                          "folds": sorted({identity(p)["fold"] for p in paths})}, indent=2))
        return
    ledger = rows()
    if args.command == "status":
        print(json.dumps({"total": len(ledger), "completed": sum(r["status"] == "COMPLETED" for r in ledger),
                          "failed": sum(r["status"] == "FAILED" for r in ledger),
                          "running": sum(r["status"] == "RUNNING" for r in ledger)}, indent=2))
        return
    with (ROOT / "launcher/worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path in paths:
            run_one(path, ledger)


if __name__ == "__main__":
    main()
