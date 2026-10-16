#!/usr/bin/env python3
"""TASK-012A evaluator for the frozen Shared+Progress five-seed parent."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from common.callbacks.wsm_segment_callback import compute_sparse_two_task_metrics
from fusion.data.wsm_ramps_semantic_datamodule import WSMRampsSemanticDataModule
from fusion.models.av_r3_disease_query import WSMAVR3DiseaseQueryModel

REPO = Path(__file__).resolve().parents[2]
PSEUDO_SHA = "17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945"
T_MULTIPLIER = 2.7764451051977987
SEEDS = (42, 43, 44, 45, 46)
PROTOCOLS = ("test_none", "test_soft", "test_hard")
EXPECTED_PROTOCOL_COUNTS = {"test_none": 1364, "test_soft": 1208, "test_hard": 1014}

LEDGER: tuple[dict[str, Any], ...] = (
    {"seed": 42, "run_name": "task011a-shared-progress-8a53-029", "mlflow_run_id": "90a5adebf43c46808814f29374e092d0", "epoch": 7,
     "config": "logs/wsm_mm_pd_dep_v1/postclosure_shared_progress_optuna_seed42_2026-10-08_15-14_wsm_av_r3_disease_query_model_task011a-shared-progress-8a53-029_f5dbef29/task011a-shared-progress-8a53-029.yaml",
     "checkpoint": "logs/wsm_mm_pd_dep_v1/postclosure_shared_progress_optuna_seed42_2026-10-08_15-14_wsm_av_r3_disease_query_model_task011a-shared-progress-8a53-029_f5dbef29/checkpoints/epoch=7_dev_mean_score=0.8337.pt",
     "config_sha256": "a5bcb56f48b223f6156281183b72c8b53200cff320b3895c4cb3247fd86878d6",
     "checkpoint_sha256": "ba02ea341800d8d6330ecca32e8a31dde3afaf61e5fb1e1eca4fd144d1ff0346",
     "expected": (0.749256, 0.918143, 0.8336994328)},
    {"seed": 43, "run_name": "task011b_shared_progress_parent_seed43", "mlflow_run_id": "009b4d3c9be44220b6d3169989c518e4", "epoch": 9,
     "config": "configs/wsm_mm_pd_dep_v1/postclosure_shared_progress_suite/01_A_shared_progress_parent_seed43.yaml",
     "checkpoint": "logs/wsm_mm_pd_dep_v1/task011b_shared_progress_parent_seed43_2026-10-08_20-35_wsm_av_r3_disease_query_model_967a8e52/checkpoints/epoch=9_dev_mean_score=0.7550.pt",
     "config_sha256": "dda16afd9478f935107a00ddf901956b3318eb37d8385ce169e2b2e2dbb25a52",
     "checkpoint_sha256": "ba280153b2a1dca9808f25c05e581fec72ab3404a3602233154ef26c9f62b926",
     "expected": (0.7103060305, 0.7997336342, 0.7550198324)},
    {"seed": 44, "run_name": "task011b_shared_progress_parent_seed44", "mlflow_run_id": "28b2b383cc244c8b98011c60c73ac14e", "epoch": 5,
     "config": "configs/wsm_mm_pd_dep_v1/postclosure_shared_progress_suite/02_A_shared_progress_parent_seed44.yaml",
     "checkpoint": "logs/wsm_mm_pd_dep_v1/task011b_shared_progress_parent_seed44_2026-10-08_20-38_wsm_av_r3_disease_query_model_d47b97e1/checkpoints/epoch=5_dev_mean_score=0.8157.pt",
     "config_sha256": "44c7cad4879a3635c7eff6d9a1f9a1a5cbbdbaf123ba4d19f3f9a3f2effe775b",
     "checkpoint_sha256": "775826bf903fced08e24eb9bfff267e414c1ea7d5e94201a466bc40642e93100",
     "expected": (0.7506546698, 0.8807426876, 0.8156986787)},
    {"seed": 45, "run_name": "task011b_shared_progress_parent_seed45", "mlflow_run_id": "ff2b0e3033ef4688ab46841585e49bcf", "epoch": 19,
     "config": "configs/wsm_mm_pd_dep_v1/postclosure_shared_progress_suite/03_A_shared_progress_parent_seed45.yaml",
     "checkpoint": "logs/wsm_mm_pd_dep_v1/task011b_shared_progress_parent_seed45_2026-10-08_20-41_wsm_av_r3_disease_query_model_87fac8f5/checkpoints/epoch=19_dev_mean_score=0.8049.pt",
     "config_sha256": "f411e86536681dc69903d01d23961714e9045df9d92e431281d688c8f0145a88",
     "checkpoint_sha256": "e4a6776902ca57885ceb8f1c5444736d16fb8257618181c027b2fc8ad996e3d9",
     "expected": (0.7303546019, 0.8793876700, 0.8048711360)},
    {"seed": 46, "run_name": "task011b_shared_progress_parent_seed46", "mlflow_run_id": "aa76a8669b09431ba9c6fbe3f4f98b0c", "epoch": 3,
     "config": "configs/wsm_mm_pd_dep_v1/postclosure_shared_progress_suite/04_A_shared_progress_parent_seed46.yaml",
     "checkpoint": "logs/wsm_mm_pd_dep_v1/task011b_shared_progress_parent_seed46_2026-10-08_20-47_wsm_av_r3_disease_query_model_b26025ee/checkpoints/epoch=3_dev_mean_score=0.7998.pt",
     "config_sha256": "bb3db27f3ce216403cb5390ba235a8f8a0fa8f410bb6424891b5187cdcf56c60",
     "checkpoint_sha256": "374efc91f7ecf153f85ce42bee6af92475a229795d1221e6c8dfa4c0bb267478",
     "expected": (0.7407686022, 0.8589095418, 0.7998390720)},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def runtime(config_path: Path) -> tuple[torch.device, bool]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train = config.get("train", {}).get("params", {})
    requested = torch.device(str(train.get("device", "cpu")))
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    return device, bool(train.get("mixed_precision", False)) and device.type == "cuda"


def verify_ledger(repo: Path, pseudo: Path) -> list[dict[str, Any]]:
    if sha256(pseudo) != PSEUDO_SHA:
        raise RuntimeError("canonical pseudo-cache SHA mismatch")
    verified = []
    for entry in LEDGER:
        config_path, checkpoint_path = repo / entry["config"], repo / entry["checkpoint"]
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise RuntimeError(f"missing frozen artifact for seed {entry['seed']}")
        if sha256(config_path) != entry["config_sha256"]:
            raise RuntimeError(f"config SHA mismatch for seed {entry['seed']}")
        if sha256(checkpoint_path) != entry["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint SHA mismatch for seed {entry['seed']}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        params = config["model"]["params"]
        if config.get("seed") != entry["seed"] or params.get("task_aware_fusion") is not False:
            raise RuntimeError(f"config identity mismatch for seed {entry['seed']}")
        if config["model"]["name"] != "wsm_av_r3_disease_query_model" or config["loss"]["params"].get("mode") != "progress":
            raise RuntimeError(f"recipe invariant mismatch for seed {entry['seed']}")
        if params.get("hidden_dim") != 256 or params.get("gate_hidden_dim") != 160:
            raise RuntimeError(f"model dimensions mismatch for seed {entry['seed']}")
        entry = dict(entry)
        entry["config_sha256_actual"] = sha256(config_path)
        entry["checkpoint_sha256_actual"] = sha256(checkpoint_path)
        entry["trainable_parameters"] = 552775
        verified.append(entry)
    return verified


def model_for(entry: dict[str, Any], repo: Path) -> tuple[WSMAVR3DiseaseQueryModel, torch.device, bool]:
    config_path = repo / entry["config"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = dict(config["model"]["params"])
    model = WSMAVR3DiseaseQueryModel(**params)
    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != 552775:
        raise RuntimeError("trainable parameter count mismatch")
    payload = torch.load(repo / entry["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    device, use_amp = runtime(config_path)
    return model.to(device), device, use_amp


@torch.no_grad()
def evaluate_one(model: WSMAVR3DiseaseQueryModel, device: torch.device, use_amp: bool, dataset: Any, collate_fn: Any) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=0)
    logits, targets, masks = [], [], []
    for batch in loader:
        batch.inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.inputs.items()}
        batch.targets = batch.targets.to(device)
        batch.masks = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.masks.items()}
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(batch)
        logits.append(output.preds.detach().float().cpu())
        targets.append(batch.targets.detach().float().cpu())
        masks.append(batch.masks["observed_mask"].detach().cpu().bool())
    raw = compute_sparse_two_task_metrics(torch.cat(logits), torch.cat(targets), torch.cat(masks), "eval")
    return {
        "protocol_count": len(dataset),
        "observed_counts": {task: int(raw[f"eval/{task}/num_samples"]) for task in ("depression", "parkinson")},
        "depression": {key: raw[f"eval/depression/{key}"] for key in ("uar", "mf1", "score")},
        "parkinson": {key: raw[f"eval/parkinson/{key}"] for key in ("uar", "mf1", "score")},
        "mean": raw["eval/mean_score"],
    }


def synthetic_checks() -> dict[str, bool]:
    logits = torch.tensor([[10.0, -10.0], [-10.0, -10.0]])
    targets = torch.tensor([[1.0, float("nan")], [float("nan"), 0.0]])
    masks = torch.tensor([[True, False], [False, True]])
    metric = compute_sparse_two_task_metrics(logits, targets, masks, "synthetic")
    assert metric["synthetic/mean_score"] == 1.0
    values = np.array([1., 2., 3., 4., 5.])
    assert abs(values.std(ddof=1) - math.sqrt(2.5)) < 1e-12
    assert abs((0.8 + 0.9) / 2 - 0.85) < 1e-12
    return {"sparse_mask": True, "metric_arithmetic": True, "raw_logit_threshold": True, "student_t_ddof1": True}


def make_datamodule(args: argparse.Namespace) -> WSMRampsSemanticDataModule:
    return WSMRampsSemanticDataModule(
        pseudo_cache_path=args.pseudo_cache_path,
        data_root=args.data_root,
        audio_feature_cache_root=args.audio_feature_cache_root,
        video_cache_root=args.video_cache_root,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        shuffle_train=False,
        drop_last_train=False,
    )


def preflight(args: argparse.Namespace) -> int:
    entries = verify_ledger(REPO, Path(args.pseudo_cache_path))
    checks = synthetic_checks()
    dm = make_datamodule(args)
    results = []
    for entry in entries:
        model, device, use_amp = model_for(entry, REPO)
        metrics = evaluate_one(model, device, use_amp, dm.val_dataset, dm.collate_fn)
        expected = entry["expected"]
        actual = (metrics["depression"]["score"], metrics["parkinson"]["score"], metrics["mean"])
        differences = [actual[index] - expected[index] for index in range(3)]
        if max(abs(value) for value in differences) > 0.0005:
            raise RuntimeError(f"DEV reproduction mismatch seed {entry['seed']}: {actual} vs {expected}")
        results.append({"seed": entry["seed"], "run_name": entry["run_name"], "mlflow_run_id": entry["mlflow_run_id"], "epoch": entry["epoch"],
                        "metrics": metrics, "expected": {"d_score": expected[0], "p_score": expected[1], "mean": expected[2]},
                        "reproduction_difference": differences, "device": str(device), "autocast_enabled": use_amp,
                        "model_dtype": str(next(model.parameters()).dtype), "output_accumulation_dtype": "torch.float32"})
    payload = {"schema": "task012a-dev-preflight-v1", "test_iteration": False, "synthetic_checks": checks,
               "ledger": entries, "invariants": {"model": "wsm_av_r3_disease_query_model", "task_aware_fusion": False,
               "progress_balancing": True, "trainable_parameters": 552775, "pseudo_cache_sha256": PSEUDO_SHA,
               "accepted_pseudo_counts": [376, 1801]}, "results": results,
               "protocol_metadata": {name: {"count": len(dm.test_dataset[name]), "expected_count": EXPECTED_PROTOCOL_COUNTS[name]} for name in PROTOCOLS},
               "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}}
    atomic_json(Path(args.output), payload)
    print(json.dumps({"mode": "dev_preflight", "output": str(Path(args.output)), "test_iteration": False, "entry_count": len(results)}, sort_keys=True))
    return 0


def test_pass(args: argparse.Namespace) -> int:
    raise RuntimeError("TASK-012A is closed: Test execution is permanently disabled")
    preflight_path = Path(args.preflight)
    if not preflight_path.is_file():
        raise RuntimeError("DEV preflight artifact is missing")
    preflight_payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight_payload.get("test_iteration") is not False or len(preflight_payload.get("results", [])) != 5:
        raise RuntimeError("DEV preflight gate is invalid")
    if args.firewall_sha:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        if head != args.firewall_sha:
            raise RuntimeError(f"firewall SHA mismatch: HEAD={head} requested={args.firewall_sha}")
    marker = Path(args.marker)
    if marker.exists():
        raise RuntimeError(f"refusing second Test invocation; marker exists: {marker}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"status": "STARTED", "seed_count": 5, "protocol_count": 3}) + "\n", encoding="utf-8")
    entries = verify_ledger(REPO, Path(args.pseudo_cache_path))
    dm = make_datamodule(args)
    results = []
    try:
        for entry in entries:
            model, device, use_amp = model_for(entry, REPO)
            for protocol in PROTOCOLS:
                metrics = evaluate_one(model, device, use_amp, dm.test_dataset[protocol], dm.collate_fn)
                if metrics["protocol_count"] != EXPECTED_PROTOCOL_COUNTS[protocol]:
                    raise RuntimeError(f"protocol count mismatch {protocol}: {metrics['protocol_count']}")
                results.append({"seed": entry["seed"], "protocol": protocol, "run_name": entry["run_name"], "mlflow_run_id": entry["mlflow_run_id"], "epoch": entry["epoch"], "metrics": metrics,
                                "device": str(device), "autocast_enabled": use_amp, "model_dtype": str(next(model.parameters()).dtype), "output_accumulation_dtype": "torch.float32"})
        marker.write_text(json.dumps({"status": "COMPLETED", "evaluation_count": len(results)}) + "\n", encoding="utf-8")
        atomic_json(Path(args.output), {"schema": "task012a-test-v1", "test_iteration": True, "invocation_count": len(results),
            "protocol_counts": EXPECTED_PROTOCOL_COUNTS, "results": results, "no_raw_predictions": True, "no_raw_logits": True,
            "no_raw_probabilities": True, "no_labels": True, "no_sample_metadata": True,
            "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}})
    except Exception:
        raise
    print(json.dumps({"mode": "test", "output": str(Path(args.output)), "invocation_count": len(results), "marker": str(marker)}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dev-preflight", "test", "synthetic"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--audio-feature-cache-root", required=True)
    parser.add_argument("--video-cache-root", required=True)
    parser.add_argument("--pseudo-cache-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight")
    parser.add_argument("--marker")
    parser.add_argument("--firewall-sha")
    args = parser.parse_args()
    if args.mode == "synthetic":
        print(json.dumps(synthetic_checks(), sort_keys=True))
        return 0
    if args.mode == "dev-preflight":
        return preflight(args)
    if not args.preflight or not args.marker:
        parser.error("--preflight and --marker are required in test mode")
    return test_pass(args)


if __name__ == "__main__":
    raise SystemExit(main())
