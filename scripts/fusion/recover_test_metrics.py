#!/usr/bin/env python3
"""Recover descriptive TEST metrics from already selected fusion checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

import chimera_plugin
from chimera_ml.core.config import ExperimentConfig
from chimera_ml.training.builders import (
    BuildContext,
    build_datamodule,
    build_loss,
    build_metrics,
    build_model,
    build_train_config,
)
from chimera_ml.training.trainer import Trainer
from chimera_ml.utils.seed import define_seed

from collect_results import checkpoint, parse_config, runtime_candidates, valid_recovery


METRICS = (
    "macro_f1",
    "weighted_f1",
    "macro_recall",
    "weighted_recall",
    "patient_recall",
    "healthy_recall",
)


def selected_test_loaders(datamodule: Any) -> dict[str, Any]:
    """Evaluate only fixed TEST populations, never VAL or TRAIN."""
    loaders = datamodule.val_dataloader()
    if not isinstance(loaders, dict) or not all(
        key in loaders for key in ("test_care", "test_wsm")
    ):
        raise ValueError("missing corpus-specific TEST loaders in Chimera DataModule")
    return {key: loaders[key] for key in ("test_care", "test_wsm")}


def evaluate(
    config_path: Path,
    checkpoint_path: Path,
    selected_epoch: int,
    selected_monitor: float,
) -> dict[str, Any]:
    config = ExperimentConfig.from_yaml(str(config_path))
    define_seed(int(config.get("seed", 0)))
    context = BuildContext(config=config, stage="eval")
    datamodule = build_datamodule(config.section("data"), context=context)
    model = build_model(config.section("model"), context=context)
    loss = build_loss(config.section("loss"), context=context)
    metrics = build_metrics(config.get("metrics", []), context=context)
    train_config = build_train_config(config.section("train"))
    train_config.epochs = 1
    trainer = Trainer(
        model=model,
        loss_fn=loss,
        optimizer=None,
        metrics=metrics,
        config=train_config,
        callbacks=[],
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    trainer.model.load_state_dict(payload.get("model_state_dict", payload), strict=True)
    results = trainer.evaluate(selected_test_loaders(datamodule))
    values = {
        f"test_{corpus}_{task}_{metric}": results[
            f"test_{corpus}/{task}/{metric}"
        ]
        for corpus in ("care", "wsm")
        for task in ("depression", "parkinson")
        for metric in METRICS
    }
    identity = parse_config(config_path)
    return {
        "run_name": identity["run_name"],
        "model": identity["model"],
        "fold": identity["fold"],
        "config_sha256": identity["config_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "selected_epoch": selected_epoch,
        "selected_val_monitor": selected_monitor,
        "provenance": "post-training inference from the original selected checkpoint",
        **values,
    }


def atomic_write_jsonl(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(
            "".join(
                json.dumps(records[name], sort_keys=True) + "\n"
                for name in sorted(records)
            )
        )
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", choices=("sweep",), default="sweep")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/fusion/sweep/recovery/test_results.jsonl"),
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    chimera_plugin.register()
    root = Path.cwd()
    configs = sorted((root / "configs" / "fusion" / args.matrix).glob("*.yaml"))
    existing = {}
    if args.output.is_file():
        for line in args.output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                run_name = row.get("run_name")
                if not isinstance(run_name, str) or run_name in existing:
                    raise ValueError(
                        f"missing/duplicate run identity in recovery file: {run_name!r}"
                    )
                existing[run_name] = row
    for path in configs[: args.limit]:
        identity = parse_config(path)
        candidates = runtime_candidates(root, identity["run_name"])
        if len(candidates) != 1:
            print(f"SKIP {identity['run_name']}: expected one runtime, found {len(candidates)}")
            continue
        selected, selected_epoch, selected_monitor, error = checkpoint(candidates[0])
        if error or selected is None:
            print(f"SKIP {identity['run_name']}: {error}")
            continue
        previous = existing.get(identity["run_name"])
        if previous is not None:
            if valid_recovery(
                previous, identity, selected, selected_epoch, selected_monitor
            ):
                print(f"RESUME {identity['run_name']}")
                continue
            raise ValueError(
                f"existing TEST recovery has invalid or stale identity/metrics: "
                f"{identity['run_name']}; refusing to overwrite"
            )
        existing[identity["run_name"]] = evaluate(
            path, selected, selected_epoch, selected_monitor
        )
        atomic_write_jsonl(args.output, existing)
        print(f"DONE {identity['run_name']}")
    print(f"recovered={len(existing)} output={args.output}")


if __name__ == "__main__":
    main()
