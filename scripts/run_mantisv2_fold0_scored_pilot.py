#!/usr/bin/env python3
"""Train one masked reference classifier on frozen MantisV2 fold-0 features."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
BATCH_SIZE = 32
EPOCHS = 20


def load_temporalizer() -> Any:
    path = Path(__file__).with_name("smoke_mantisv2_egemaps_train.py")
    spec = importlib.util.spec_from_file_location("egemaps_temporalizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load temporalizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def sentinel_check(temporalizer: Any, channels: list[str]) -> None:
    frames = []
    for index in range(100):
        start = index * temporalizer.NOMINAL_HOP_SECONDS
        frames.append(
            {
                "turn_id": "sentinel",
                "turn_start_sec": 0.0,
                "turn_end_sec": 1.0,
                "frame_start_sec": start,
                "frame_end_sec": start + 0.02,
                "values": np.full(len(channels), index, dtype=np.float32),
            }
        )
    excluded = dict(frames[-1], frame_end_sec=1.0000625)
    retained = frames[:-1]
    baseline = temporalizer.temporalize(retained, len(channels))[0]["tensor"]
    contaminated = temporalizer.temporalize(frames, len(channels))[0]["tensor"]
    assert np.array_equal(baseline, contaminated)
    assert float(excluded["values"].max()) == 99.0


def build_population(
    corpus: str,
    split: str,
    root: Path,
    manifest: dict[tuple[str, str], dict[str, str]],
    temporalizer: Any,
    channels: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = read_csv(root / "splits" / "folds" / f"fold_0_{split}.csv")
    records: list[dict[str, Any]] = []
    summary = {
        "source_records": len(rows),
        "manifest_valid_records": 0,
        "valid_records_with_windows": 0,
        "records_without_valid_window": 0,
        "policy_c_excluded_frames": 0,
        "accepted_windows": 0,
        "source_frame_rows": 0,
    }
    for row in rows:
        record = manifest.get((corpus, row["sample_id"]))
        if record is None or record.get("egemaps_lld_status") != "VALID":
            continue
        summary["manifest_valid_records"] += 1
        path = temporalizer.artifact_path(root, row)
        diagnostic = temporalizer.inspect_artifact(
            path,
            channels,
            temporalizer.NOMINAL_HOP_SECONDS,
            corpus,
            "terminal-exclude",
        )
        if diagnostic["status"] != "VALID":
            summary["records_without_valid_window"] += 1
            continue
        frames, excluded, reasons = temporalizer.artifact_frames(
            path, channels, temporalizer
        )
        windows = temporalizer.temporalize(frames, len(channels))
        summary["policy_c_excluded_frames"] += excluded
        summary["accepted_windows"] += len(windows)
        summary["source_frame_rows"] += diagnostic["original_rows"]
        if not windows:
            summary["records_without_valid_window"] += 1
            continue
        summary["valid_records_with_windows"] += 1
        labels = np.asarray(
            [float(record["depression"]), float(record["parkinson"])],
            dtype=np.float32,
        )
        observed = np.asarray(
            [record["depression_observed"] == "True", record["parkinson_observed"] == "True"],
            dtype=bool,
        )
        records.append(
            {
                "corpus": corpus,
                "labels": labels,
                "observed": observed,
                "windows": windows,
                "excluded": excluded,
                "reasons": reasons,
            }
        )
    return records, summary


def infer_records(
    records: list[dict[str, Any]],
    network: nn.Module,
    trainer: Any,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    representations = []
    metadata = []
    for index in range(0, len(records), batch_size):
        group = records[index : index + batch_size]
        window_tensors = [window["tensor"] for record in group for window in record["windows"]]
        inputs = torch.from_numpy(np.stack(window_tensors)).to(device=device, dtype=torch.float32)
        assert inputs.shape[1:] == (25, 512)
        with torch.no_grad():
            embeddings = trainer.transform(inputs, three_dim=True, to_numpy=False)
        assert embeddings.shape[1:] == (25, 512)
        offset = 0
        for record in group:
            count = len(record["windows"])
            pooled = embeddings[offset : offset + count].mean(dim=0).cpu().numpy()
            representations.append(pooled.reshape(-1).astype(np.float32))
            for window in record["windows"]:
                metadata.append(
                    {
                        "corpus": record["corpus"],
                        "window_start_sec": window["start_sec"],
                        "window_end_sec": window["end_sec"],
                        "source_frame_count": window["source_frame_count"],
                        "source_start_sec": window["source_start_sec"],
                        "source_end_sec": window["source_end_sec"],
                        "policy_c_excluded_frames": record["excluded"],
                        "policy_c_exclusion_reasons": record["reasons"],
                    }
                )
            offset += count
    labels = np.stack([record["labels"] for record in records]).astype(np.float32)
    observed = np.stack([record["observed"] for record in records])
    return np.stack(representations), labels, observed, metadata


def metric_values(logits: torch.Tensor, targets: torch.Tensor, observed: torch.Tensor) -> dict[str, Any]:
    result = {}
    for name, task in (("depression", 0), ("parkinson", 1)):
        mask = observed[:, task].bool()
        target = targets[mask, task].int()
        prediction = (logits[mask, task] > 0).int()
        n = int(mask.sum())
        positive = int((target == 1).sum())
        negative = int((target == 0).sum())
        prefix = f"{name}"
        result[f"{prefix}_support"] = {"observed": n, "positive": positive, "negative": negative}
        if not n or not positive or not negative:
            for metric in ("macro_f1", "weighted_f1", "macro_recall", "weighted_recall", "patient_recall", "healthy_recall"):
                result[f"{prefix}_{metric}"] = None
            continue
        tp = int(((prediction == 1) & (target == 1)).sum())
        tn = int(((prediction == 0) & (target == 0)).sum())
        fp = int(((prediction == 1) & (target == 0)).sum())
        fn = int(((prediction == 0) & (target == 1)).sum())
        patient_recall = tp / positive
        healthy_recall = tn / negative
        patient_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        healthy_f1 = 2 * tn / max(2 * tn + fp + fn, 1)
        result.update(
            {
                f"{prefix}_macro_f1": (patient_f1 + healthy_f1) / 2,
                f"{prefix}_weighted_f1": (patient_f1 * positive + healthy_f1 * negative) / n,
                f"{prefix}_macro_recall": (patient_recall + healthy_recall) / 2,
                f"{prefix}_weighted_recall": (patient_recall * positive + healthy_recall * negative) / n,
                f"{prefix}_patient_recall": patient_recall,
                f"{prefix}_healthy_recall": healthy_recall,
            }
        )
    components = [result.get(f"{task}_macro_recall") for task in ("depression", "parkinson")]
    result["macro_recall"] = None if any(value is None for value in components) else sum(components) / 2
    return result


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, observed: torch.Tensor, device: str) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
    return metric_values(logits.cpu(), y, observed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("logs/mantisv2/fold0_scored_pilot"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/feature_audit/sample_feature_availability.csv"))
    parser.add_argument("--care-root", type=Path, default=Path("/media/maxim/Databases/CARE"))
    parser.add_argument("--wsm-root", type=Path, default=Path("/media/maxim/Databases/WSM_NEW"))
    parser.add_argument("--registry", type=Path, default=Path("/media/maxim/Databases/CARE/splits/features.json"))
    args = parser.parse_args()
    seed_everything()
    temporalizer = load_temporalizer()
    channels = temporalizer.registry_channels(args.registry)
    sentinel_check(temporalizer, channels)
    manifest = {(row["corpus"], row["sample_id"]): row for row in read_csv(args.manifest)}
    populations = {}
    summaries = {}
    for corpus, root in (("CARE", args.care_root), ("WSM", args.wsm_root)):
        for split in ("train", "val"):
            key = f"{split}_{corpus.lower()}"
            populations[key], summaries[key] = build_population(
                corpus, split, root, manifest, temporalizer, channels
            )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer

    network = MantisV2(device=device, return_transf_layer=2, output_token="combined")
    network = network.from_pretrained(str(args.checkpoint))
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    trainer = MantisTrainer(device=device, network=network)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, Any] = {"checkpoint_revision": "99fe0f548960e272fbfa4b82fd9b5b5956779dfd", "channels": channels, "populations": summaries}
    arrays = {}
    for key, records in populations.items():
        x, y, observed, metadata = infer_records(records, network, trainer, device, 4)
        arrays[key] = (x, y, observed)
        cache[key] = {"records": len(records), "metadata": metadata}
        np.savez_compressed(args.cache_dir / f"{key}.npz", x=x, y=y, observed=observed)
    (args.cache_dir / "metadata.json").write_text(json.dumps(cache, indent=2), encoding="utf-8")
    train_x = np.concatenate([arrays["train_care"][0], arrays["train_wsm"][0]])
    train_y = np.concatenate([arrays["train_care"][1], arrays["train_wsm"][1]])
    train_observed = np.concatenate([arrays["train_care"][2], arrays["train_wsm"][2]])
    val_x = np.concatenate([arrays["val_care"][0], arrays["val_wsm"][0]])
    val_y = np.concatenate([arrays["val_care"][1], arrays["val_wsm"][1]])
    val_observed = np.concatenate([arrays["val_care"][2], arrays["val_wsm"][2]])
    train_tensor = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_observed))
    loader = DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    pos_weights = []
    for task in range(2):
        known = train_observed[:, task]
        pos = train_y[known, task].sum()
        neg = known.sum() - pos
        pos_weights.append(float(neg / pos))
    model = nn.Sequential(nn.Linear(12800, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    weight = torch.tensor(pos_weights, dtype=torch.float32, device=device)
    train_x_t, train_y_t, train_o_t = train_tensor.tensors
    val_x_t, val_y_t, val_o_t = torch.from_numpy(val_x), torch.from_numpy(val_y), torch.from_numpy(val_observed)
    best = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch_x, batch_y, batch_observed in loader:
            optimizer.zero_grad()
            logits = model(batch_x.to(device))
            task_losses = []
            for task in range(2):
                mask = batch_observed[:, task].bool().to(device)
                task_losses.append(F.binary_cross_entropy_with_logits(logits[mask, task], batch_y[mask, task].to(device), pos_weight=weight[task]))
            loss = torch.stack(task_losses).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate(model, val_x_t, val_y_t, val_o_t, device)
        score = val_metrics["macro_recall"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val_metrics})
        if score is not None and (best is None or score > best["score"]):
            best = {"epoch": epoch, "score": score, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    assert best is not None
    model.load_state_dict(best["state"])
    final_metrics = {}
    for corpus in ("care", "wsm"):
        final_metrics[corpus] = evaluate(model, torch.from_numpy(arrays[f"val_{corpus}"][0]), torch.from_numpy(arrays[f"val_{corpus}"][1]), torch.from_numpy(arrays[f"val_{corpus}"][2]), device)
    result = {"device": device, "training": {"seed": SEED, "epochs_max": EPOCHS, "best_epoch": best["epoch"], "pos_weights": pos_weights, "history": history}, "validation": final_metrics, "fold0_macro_recall": None if any(final_metrics[c]["macro_recall"] is None for c in ("care", "wsm")) else (final_metrics["care"]["macro_recall"] + final_metrics["wsm"]["macro_recall"]) / 2, "coverage": summaries}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
