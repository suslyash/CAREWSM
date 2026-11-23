"""Generate the committed AV-005 285-configuration sweep matrix."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "configs/fusion/sweep"
SOURCES = {
    "audio": {"audio_id1": 64, "audio_id2": 64, "audio_id4": 64},
    "video": {"video_4": 128},
    "text": {"text_id9": 64, "text_id10": 64},
}
BIMODAL_MODELS = (
    ("bimodal_gated_late", "carewsm_bimodal_gated_late"),
    ("bimodal_shared_mtl", "carewsm_bimodal_shared_mtl"),
    ("bimodal_task_aware", "carewsm_bimodal_task_aware"),
)
TRIMODAL_MODELS = (
    ("text_guided", "carewsm_text_guided_dual_gated"),
    ("audio_guided", "carewsm_audio_guided_dual_gated"),
    ("cross_attention", "carewsm_trimodal_cross_attention"),
    ("shared_private", "carewsm_shared_private_trimodal"),
)


def config(
    model: str,
    modalities: tuple[str, ...],
    selected: dict[str, str],
    fold: int,
    run: str,
) -> dict:
    dimensions = {
        modality: SOURCES[modality][selected[modality]] for modality in modalities
    }
    return {
        "callbacks": [
            {"name": "carewsm_validation_monitor", "params": {}},
            {
                "name": "checkpoint_callback",
                "params": {
                    "filename_template": "epoch={epoch}_{monitor}={value:.4f}.pt",
                    "log_path": "logs/fusion/sweep",
                    "experiment_name": "sweep",
                    "run_name": run,
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
            {
                "name": "snapshot_callback",
                "params": {
                    "include": ["src", "configs"],
                    "log_path": "logs/fusion/sweep",
                    "experiment_name": "sweep",
                    "run_name": run,
                    "save_code_zip": True,
                    "save_config": True,
                },
            },
        ],
        "data": {
            "name": "carewsm_fusion",
            "params": {
                "care_root": "/media/maxim/Databases/CARE",
                "wsm_root": "/media/maxim/Databases/WSM_NEW",
                "manifest": "reports/feature_audit/sample_feature_availability.csv",
                "fold": fold,
                "modalities": list(modalities),
                "audio_source": selected.get("audio", "audio_id1"),
                "video_source": selected.get("video", "video_4"),
                "text_source": selected.get("text", "text_id9"),
                "representation_key": "shared_features",
                "batch_size": 32,
                "num_workers": 8,
                "shuffle_train": True,
            },
        },
        "model": {
            "name": model,
            "params": {
                "modalities": list(modalities),
                "dimensions": dimensions,
                "hidden_dim": 64,
                "dropout": 0.1,
            },
        },
        "loss": {"name": "carewsm_masked_bce"},
        "metrics": [{"name": "carewsm_masked_multitask_classification"}],
        "optimizer": {
            "name": "adamw_optimizer",
            "params": {"lr": 0.0001, "weight_decay": 0.0},
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
        "experiment_info": {
            "params": {
                "experiment_name": "sweep",
                "run_name": run,
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
                    "log_path": "logs/fusion/sweep",
                    "experiment_name": "sweep",
                    "run_name": run,
                },
            },
            {
                "name": "mlflow_logger",
                "params": {
                    "tracking_uri": "sqlite:///logs/fusion/sweep/mlflow.db",
                    "experiment_name": "sweep",
                    "run_name": run,
                },
            },
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.glob("*.yaml"):
        path.unlink()
    identities = []
    pairs = (("audio", "video"), ("audio", "text"), ("video", "text"))
    for model_token, model in BIMODAL_MODELS:
        for modalities in pairs:
            for selected_values in product(*(tuple(SOURCES[m]) for m in modalities)):
                selected = dict(zip(modalities, selected_values))
                feature_token = "_".join(
                    source.removeprefix("audio_id")
                    .removeprefix("video_")
                    .removeprefix("text_id")
                    for source in selected_values
                )
                feature_token = "_".join(
                    (
                        "A" + source.removeprefix("audio_id")
                        if m == "audio"
                        else (
                            "V" + source.removeprefix("video_")
                            if m == "video"
                            else "T" + source.removeprefix("text_id")
                        )
                    )
                    for m, source in zip(modalities, selected_values)
                )
                for fold in range(5):
                    run = f"{model_token}__{feature_token}__fold{fold}"
                    (OUT / f"{run}.yaml").write_text(
                        yaml.safe_dump(
                            config(model, modalities, selected, fold, run),
                            sort_keys=False,
                        )
                    )
                    identities.append(run)
    for model_token, model in TRIMODAL_MODELS:
        for audio, text in product(SOURCES["audio"], SOURCES["text"]):
            selected = {"audio": audio, "video": "video_4", "text": text}
            feature_token = (
                f"A{audio.removeprefix('audio_id')}_V4_T{text.removeprefix('text_id')}"
            )
            for fold in range(5):
                run = f"{model_token}__{feature_token}__fold{fold}"
                (OUT / f"{run}.yaml").write_text(
                    yaml.safe_dump(
                        config(model, ("audio", "video", "text"), selected, fold, run),
                        sort_keys=False,
                    )
                )
                identities.append(run)
    assert len(identities) == len(set(identities)) == 285
    print(f"generated {len(identities)} unique AV-005 configurations")


if __name__ == "__main__":
    main()
