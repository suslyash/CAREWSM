#!/usr/bin/env python3
"""Create the isolated 20-config turnless mean-pooling experiment matrix."""

from __future__ import annotations

from pathlib import Path

import yaml


WEIGHTS = [
    2.358878504672897,
    2.375234521575985,
    2.3653308480894686,
    2.4134980988593155,
    2.3594771241830066,
]
MODELS = {
    "E1_gated64": ("carewsm_gated_fusion", 64),
    "E2_mlp64": ("carewsm_pooled_mlp", 64),
    "E3_gated32": ("carewsm_gated_fusion", 32),
    "E4_mlp32": ("carewsm_pooled_mlp", 32),
}


def main() -> None:
    root = Path("configs/audio_representation_pilot/turnless_mean")
    root.mkdir(parents=True, exist_ok=True)
    for experiment, (model, hidden_dim) in MODELS.items():
        for fold, weight in enumerate(WEIGHTS):
            config = {
                "callbacks": [
                    {"name": "carewsm_validation_monitor", "params": {}},
                    {
                        "name": "checkpoint_callback",
                        "params": {
                            "filename_template": "epoch={epoch}_{monitor}={value:.4f}.pt",
                            "log_path": "logs/audio_representation_pilot/turnless_mean",
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
                    {
                        "name": "carewsm_summary_callback",
                        "params": {"filename": "summary.txt"},
                    },
                    {
                        "name": "snapshot_callback",
                        "params": {
                            "include": ["src", "configs"],
                            "log_path": "logs/audio_representation_pilot/turnless_mean",
                            "save_code_zip": True,
                            "save_config": True,
                        },
                    },
                ],
                "data": {
                    "name": "carewsm_joint",
                    "params": {
                        "batch_size": 64,
                        "care_root": "/media/maxim/Databases/CARE",
                        "feature_family": ["emotionmsp_embeds", "trillsson4_1"],
                        "fold": fold,
                        "manifest": "reports/feature_audit/sample_feature_availability.csv",
                        "num_workers": 8,
                        "pooling": "mean",
                        "registry": "/media/maxim/Databases/CARE/splits/features.json",
                        "shuffle_train": True,
                        "wsm_root": "/media/maxim/Databases/WSM_NEW",
                    },
                },
                "experiment_info": {
                    "params": {
                        "experiment_name": "audio_representation_pilot_turnless_mean",
                        "run_name": f"{experiment}_fold{fold}",
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
                            "log_path": "logs/audio_representation_pilot/turnless_mean",
                        },
                    },
                    {
                        "name": "mlflow_logger",
                        "params": {
                            "tracking_uri": "sqlite:///logs/audio_representation_pilot/turnless_mean/mlflow.db"
                        },
                    },
                ],
                "loss": {
                    "name": "carewsm_weighted_parkinson_bce",
                    "params": {"parkinson_pos_weight": weight},
                },
                "metrics": [{"name": "carewsm_masked_multitask_classification"}],
                "model": {
                    "name": model,
                    "params": {
                        "input_dim": 2048,
                        "hidden_dim": hidden_dim,
                        "dropout": 0.1,
                        **(
                            {"family_dims": [1024, 1024]}
                            if model == "carewsm_gated_fusion"
                            else {}
                        ),
                    },
                },
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
            path = root / f"{experiment}_fold{fold}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Created {len(MODELS) * 5} configurations in {root}")


if __name__ == "__main__":
    main()
