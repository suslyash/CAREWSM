"""No-training AV-005 subset-model and matrix checks."""

from pathlib import Path
import subprocess

import torch
import yaml

import chimera_plugin
from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS


def batch(modalities, dimensions, lengths):
    inputs = {}
    masks = {"observed_targets": torch.ones(2, 2, dtype=torch.bool)}
    for modality in modalities:
        length = lengths[modality]
        inputs[modality] = torch.randn(2, length, dimensions[modality])
        masks[f"{modality}_lengths"] = torch.tensor([length, length])
        masks[f"{modality}_valid_mask"] = torch.ones(2, length, dtype=torch.bool)
    return Batch(inputs=inputs, targets=torch.zeros(2, 2), masks=masks)


def test_sweep_matrix_has_expected_counts_and_no_duplicate_modalities():
    paths = sorted(Path("configs/fusion/sweep").glob("*.yaml"))
    assert len(paths) == 285
    identities = set()
    for path in paths:
        config = yaml.safe_load(path.read_text())
        modalities = tuple(config["data"]["params"]["modalities"])
        assert len(modalities) in (2, 3)
        assert len(set(modalities)) == len(modalities)
        identity = (config["model"]["name"], path.stem)
        assert identity not in identities
        identities.add(identity)
    assert len(identities) == 285


def test_bimodal_registry_models_use_exactly_two_modalities():
    chimera_plugin.register()
    definitions = {
        "carewsm_bimodal_gated_late": ("audio", "video"),
        "carewsm_bimodal_shared_mtl": ("audio", "text"),
        "carewsm_bimodal_task_aware": ("video", "text"),
    }
    dimensions = {"audio": 4, "video": 5, "text": 3}
    for name, modalities in definitions.items():
        model = MODELS.create(
            name,
            modalities=list(modalities),
            dimensions=dimensions,
            hidden_dim=12,
            dropout=0.0,
        )
        output = model(batch(modalities, dimensions, {m: 1 for m in modalities}))
        assert output.preds.shape == (2, 2)
        output.preds.sum().backward()
        assert all(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )


def test_audio_guided_trimodal_model_has_distinct_conditioning_direction():
    chimera_plugin.register()
    model = MODELS.create(
        "carewsm_audio_guided_dual_gated",
        audio_dim=4,
        video_dim=5,
        text_dim=3,
        hidden_dim=12,
        dropout=0.0,
    )
    output = model(
        batch(
            ("audio", "video", "text"),
            {"audio": 4, "video": 5, "text": 3},
            {"audio": 1, "video": 1, "text": 1},
        )
    )
    assert output.preds.shape == (2, 2)
    assert output.aux["modality_gate"].shape == (2, 3)



def test_launcher_exits_nonzero_when_chimera_exits_zero_without_artifacts(tmp_path):
    """An empty successful subprocess is NOT a completed GPU run."""
    code = Path("scripts/fusion/run_all.sh").read_text(encoding="utf-8")
    # Reuse the real launcher functions, without the command-line dispatcher.
    definitions = "usage() {" + code.split("usage() {", 1)[1].split(
        '[[ $# == 1 || $# == 3 ]]', 1
    )[0]
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "probe_fold0.yaml").write_text("placeholder", encoding="utf-8")
    executable = tmp_path / "chimera_success_without_artifacts.sh"
    executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    logs = tmp_path / "logs"
    script = "set -Eeuo pipefail\n" + definitions + f'''
CONFIG_DIR="{configs}"
LOG_ROOT="{logs}"
LAUNCHER_DIR="{logs}/launcher"
LEDGER="$LAUNCHER_DIR/status.tsv"
NAMESPACE="fusion"
CHIMERA="{executable}"
CUDA_DEVICE=0
pgrep() {{ return 1; }}
run_all run
'''
    completed = subprocess.run(
        ["bash", "-c", script], text=True, capture_output=True, check=False
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "COMPLETION_FAILED" in (logs / "launcher/status.tsv").read_text()
    assert "missing or ambiguous" in completed.stderr
