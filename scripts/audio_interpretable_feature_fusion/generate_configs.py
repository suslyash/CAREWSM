#!/usr/bin/env python3
"""Generate the bounded 25-run interpretable static-family matrix."""
from pathlib import Path
import yaml

ROOT = Path("configs/audio_interpretable_feature_fusion")
LOG_ROOT = "logs/audio_interpretable_feature_fusion"
FAMILIES = (
    "emotionmsp_embeds", "emotionmsp_trajectories", "trillsson4_1",
    "wav2vec_xlsr300m", "egemaps_func", "disvoice_phonation_static",
    "disvoice_articulation_static",
)
COHORT = list(FAMILIES)
EXPERIMENTS = {
    "B": FAMILIES[:4],
    "BE": FAMILIES[:5],
    "BP": [*FAMILIES[:4], FAMILIES[5]],
    "BA": [*FAMILIES[:4], FAMILIES[6]],
    "BEPA": list(FAMILIES),
}
DIMS = dict(zip(FAMILIES, (1024, 3, 1024, 1024, 88, 28, 488), strict=True))
WEIGHTS = [2.358878504672897, 2.375234521575985, 2.3653308480894686, 2.4134980988593155, 2.3594771241830066]

def make(name, families, fold):
    run = f"interp_{name}_mean_gated64_equal_fold{fold}"
    dims = [DIMS[family] for family in families]
    return {
        "callbacks": [
            {"name": "carewsm_validation_monitor", "params": {}},
            {"name": "checkpoint_callback", "params": {"filename_template": "epoch={epoch}_{monitor}={value:.4f}.pt", "log_path": LOG_ROOT, "mode": "max", "monitor": "val_mean_corpora_tasks/macro_recall", "save_last": True, "save_top_k": 1}},
            {"name": "early_stopping_callback", "params": {"min_delta": 0.0, "mode": "max", "monitor": "val_mean_corpora_tasks/macro_recall", "patience": 7}},
            {"name": "carewsm_summary_callback", "params": {"filename": "summary.txt"}},
            {"name": "carewsm_stageg_mtl_callback", "params": {}},
            {"name": "snapshot_callback", "params": {"include": ["src", "configs", "scripts/audio_interpretable_feature_fusion"], "log_path": LOG_ROOT, "save_code_zip": True, "save_config": True}},
        ],
        "data": {"name": "carewsm_joint", "params": {
            "batch_size": 64, "care_root": "/media/maxim/Databases/CARE", "feature_family": families,
            "cohort_feature_family": COHORT, "independent_family_mean": True, "fold": fold,
            "manifest": "reports/feature_audit/sample_feature_availability.csv", "num_workers": 8, "pooling": "mean",
            "registry": "/media/maxim/Databases/CARE/splits/features.json", "shuffle_train": True, "wsm_root": "/media/maxim/Databases/WSM_NEW"}},
        "experiment_info": {"params": {"experiment_name": "audio_interpretable_feature_fusion", "run_name": run, "include_time": False}},
        "logging": [{"name": "console_file_logger", "params": {"console_level": "INFO", "file_level": "INFO", "log_file": "train.log", "log_path": LOG_ROOT}}, {"name": "mlflow_logger", "params": {"tracking_uri": f"sqlite:///{LOG_ROOT}/mlflow.db"}}],
        "loss": {"name": "carewsm_stageg_mtl", "params": {"method": "equal", "parkinson_pos_weight": WEIGHTS[fold]}},
        "metrics": [{"name": "carewsm_masked_multitask_classification"}],
        "model": {"name": "carewsm_static_gated_fusion", "params": {"input_dim": sum(dims), "hidden_dim": 64, "dropout": 0.1, "family_dims": dims}},
        "optimizer": {"name": "adamw_optimizer", "params": {"lr": 1e-4, "weight_decay": 0.0}},
        "seed": 5, "train": {"params": {"collect_cache": True, "device": "cuda", "epochs": 50, "grad_clip_norm": 1.0, "log_every_steps": 100, "mixed_precision": True}},
    }

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, families in EXPERIMENTS.items():
        for fold in range(5):
            path=ROOT/f"interp_{name}_mean_gated64_equal_fold{fold}.yaml"
            path.write_text(yaml.safe_dump(make(name, families, fold), sort_keys=False), encoding="utf-8")

if __name__ == "__main__": main()
