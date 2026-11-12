#!/usr/bin/env python3
"""Generate only the bounded audio_turns_combined configuration stages."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

FAMILIES = {
    "A": ("emotionmsp_embeds", 1024),
    "B": ("emotionmsp_trajectories", 3),
    "C": ("trillsson4_1", 1024),
    "D": ("wav2vec_xlsr300m", 1024),
}
WEIGHTS = [
    2.358878504672897,
    2.375234521575985,
    2.3653308480894686,
    2.4134980988593155,
    2.3594771241830066,
]
ROOT = Path("configs/audio_turns_combined")


def combinations_all() -> list[str]:
    return [
        "".join(item) for size in range(1, 5) for item in combinations("ABCD", size)
    ]


def family_names(code: str) -> list[str]:
    return [FAMILIES[letter][0] for letter in code]


def dimensions(code: str, representation: str) -> list[int]:
    multiplier = 2 if representation == "mean_std" else 1
    return [FAMILIES[letter][1] * multiplier for letter in code]


def config(
    stage: str,
    experiment_id: str,
    code: str,
    representation: str,
    model: str,
    hidden_dim: int,
    fold: int,
    loss: str = "weighted",
) -> dict[str, Any]:
    families = family_names(code)
    configured_families: str | list[str] = (
        families[0] if len(families) == 1 else families
    )
    dims = dimensions(code, representation)
    if loss == "weighted":
        loss_config = {
            "name": "carewsm_weighted_parkinson_bce",
            "params": {"parkinson_pos_weight": WEIGHTS[fold]},
        }
    else:
        loss_config = {
            "name": "carewsm_stageg_mtl",
            "params": {"method": loss, "parkinson_pos_weight": WEIGHTS[fold]},
        }
    callbacks = [
        {"name": "carewsm_validation_monitor", "params": {}},
        {
            "name": "checkpoint_callback",
            "params": {
                "filename_template": "epoch={epoch}_{monitor}={value:.4f}.pt",
                "log_path": "logs/audio_turns_combined",
                "mode": "max",
                "monitor": "val_mean_corpora_tasks/macro_recall",
                "save_last": True,
                "save_top_k": 1,
            },
        },
        {
            "name": "early_stopping_callback",
            "params": {
                "min_delta": 0.0,
                "mode": "max",
                "monitor": "val_mean_corpora_tasks/macro_recall",
                "patience": 7,
            },
        },
        {"name": "carewsm_summary_callback", "params": {"filename": "summary.txt"}},
    ]
    if loss != "weighted":
        callbacks.append({"name": "carewsm_stageg_mtl_callback", "params": {}})
    callbacks.append(
        {
            "name": "snapshot_callback",
            "params": {
                "include": ["src", "configs"],
                "log_path": "logs/audio_turns_combined",
                "save_code_zip": True,
                "save_config": True,
            },
        }
    )
    model_params = {"input_dim": sum(dims), "hidden_dim": hidden_dim, "dropout": 0.1}
    if model == "carewsm_static_gated_fusion":
        model_params["family_dims"] = dims
    return {
        "callbacks": callbacks,
        "data": {
            "name": "carewsm_joint",
            "params": {
                "batch_size": 64,
                "care_root": "/media/maxim/Databases/CARE",
                "feature_family": configured_families,
                "fold": fold,
                "manifest": "reports/feature_audit/sample_feature_availability.csv",
                "num_workers": 8,
                "pooling": representation,
                "registry": "/media/maxim/Databases/CARE/splits/features.json",
                "shuffle_train": True,
                "wsm_root": "/media/maxim/Databases/WSM_NEW",
            },
        },
        "experiment_info": {
            "params": {
                "experiment_name": "audio_turns_combined",
                "run_name": f"{experiment_id}_fold{fold}",
                "include_time": False,
            }
        },
        "logging": [
            {
                "name": "console_file_logger",
                "params": {
                    "console_level": "INFO",
                    "file_level": "INFO",
                    "log_file": "train.log",
                    "log_path": "logs/audio_turns_combined",
                },
            },
            {
                "name": "mlflow_logger",
                "params": {
                    "tracking_uri": "sqlite:///logs/audio_turns_combined/mlflow.db"
                },
            },
        ],
        "loss": loss_config,
        "metrics": [{"name": "carewsm_masked_multitask_classification"}],
        "model": {"name": model, "params": model_params},
        "optimizer": {
            "name": "adamw_optimizer",
            "params": {"lr": 1e-4, "weight_decay": 0.0},
        },
        "seed": 5,
        "train": {
            "params": {
                "collect_cache": True,
                "device": "cuda",
                "epochs": 50,
                "grad_clip_norm": 1.0,
                "log_every_steps": 100,
                "mixed_precision": True,
            }
        },
    }


def write(stage: str, entries: list[tuple[str, str, str, int, str]]) -> None:
    directory = ROOT / stage
    directory.mkdir(parents=True, exist_ok=True)
    for experiment_id, code, representation, hidden_dim, model in entries:
        for fold in range(5):
            loss = "weighted"
            if stage == "mtl_strategies":
                loss = experiment_id.rsplit("_", 1)[-1]
            path = directory / f"{experiment_id}_fold{fold}.yaml"
            path.write_text(
                yaml.safe_dump(
                    config(
                        stage,
                        experiment_id,
                        code,
                        representation,
                        model,
                        hidden_dim,
                        fold,
                        loss,
                    ),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("stage1", "stage2", "stage3", "stage4", "stage5")
    )
    parser.add_argument("--entries", nargs="*", default=[])
    args = parser.parse_args()
    if args.stage == "stage1":
        write(
            "feature_screening",
            [
                (f"screen_{code}_mean_mlp32", code, "mean", 32, "carewsm_static_mlp")
                for code in combinations_all()
            ],
        )
    elif not args.entries:
        raise SystemExit(
            "dynamic stages require --entries CODE:REPRESENTATION:MODEL:HIDDEN or CODE:REPRESENTATION:MODEL:HIDDEN:STRATEGY"
        )
    elif args.stage == "stage2":
        write(
            "gated_fusion",
            [
                (
                    f"gated_{item.split(':')[0]}_mean_gated32",
                    item.split(":")[0],
                    "mean",
                    32,
                    "carewsm_static_gated_fusion",
                )
                for item in args.entries
            ],
        )
    elif args.stage == "stage3":
        write(
            "mean_std",
            [
                (
                    f"meanstd_{item.split(':')[0]}_{item.split(':')[2]}{item.split(':')[3]}",
                    item.split(":")[0],
                    "mean_std",
                    int(item.split(":")[3]),
                    item.split(":")[2],
                )
                for item in args.entries
            ],
        )
    elif args.stage == "stage4":
        write(
            "model_tuning",
            [
                (
                    f"tune_{item.split(':')[0]}_{item.split(':')[1]}{item.split(':')[2]}",
                    item.split(":")[0],
                    item.split(":")[1],
                    int(item.split(":")[2]),
                    item.split(":")[3],
                )
                for item in args.entries
            ],
        )
    else:
        write(
            "mtl_strategies",
            [
                (
                    f"mtl_{item.split(':')[0]}_{item.split(':')[1]}_{item.split(':')[2]}",
                    item.split(":")[0],
                    item.split(":")[1],
                    int(item.split(":")[2]),
                    item.split(":")[3],
                )
                for item in args.entries
            ],
        )


if __name__ == "__main__":
    main()
