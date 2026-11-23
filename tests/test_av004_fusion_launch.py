from pathlib import Path

import yaml


CONFIG_ROOT = Path("configs/fusion")
MODELS = {
    "text_guided_dual_gated": "carewsm_text_guided_dual_gated",
    "trimodal_cross_attention": "carewsm_trimodal_cross_attention",
    "shared_private_trimodal": "carewsm_shared_private_trimodal",
}


def test_fusion_matrix_and_runtime_destinations() -> None:
    paths = sorted(CONFIG_ROOT.glob("*_fold[0-4].yaml"))
    assert len(paths) == 15
    identities = set()
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        params = config["data"]["params"]
        model = config["model"]["name"]
        fold = params["fold"]
        identities.add((model, fold))
        assert model in MODELS.values()
        assert path.stem == config["experiment_info"]["params"]["run_name"]
        assert params["audio_source"] == "audio_id1"
        assert params["video_source"] == "video_4"
        assert params["text_source"] == "text_id9"
        assert config["model"]["params"] == {
            "audio_dim": 64,
            "video_dim": 128,
            "text_dim": 64,
            "hidden_dim": 64,
            "dropout": 0.1,
        }
        assert config["logging"][0]["params"]["log_path"] == "logs/fusion"
        assert config["logging"][1]["params"]["tracking_uri"] == (
            "sqlite:///logs/fusion/mlflow.db"
        )
        for callback in config["callbacks"]:
            if "log_path" in callback.get("params", {}):
                assert callback["params"]["log_path"] == "logs/fusion"
    assert len(identities) == 15
