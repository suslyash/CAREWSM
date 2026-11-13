from pathlib import Path

import yaml

from scripts.audio_turns_mean_mtl_completion.generate_configs import METHODS, make


def test_issue_109_matrix_has_exact_methods_and_folds() -> None:
    configs = [make(method, fold) for method in METHODS for fold in range(5)]
    assert len(configs) == 20
    assert {config["loss"]["params"]["method"] for config in configs} == set(METHODS)
    assert {config["data"]["params"]["fold"] for config in configs} == set(range(5))
    assert all(config["data"]["params"]["pooling"] == "mean" for config in configs)
    assert all(config["model"]["params"]["input_dim"] == 3075 for config in configs)
    assert all(config["seed"] == 5 for config in configs)


def test_issue_109_generated_yaml_preserves_fold_weights(tmp_path: Path) -> None:
    config = make("cagrad", 4)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["loss"]["params"]["parkinson_pos_weight"] == 2.3594771241830066
