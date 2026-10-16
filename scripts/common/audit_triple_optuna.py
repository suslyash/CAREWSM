"""Pre-run firewall for TASK-005H-TRIPLE-OPTUNA."""
from __future__ import annotations

import hashlib
import inspect
from itertools import product
from pathlib import Path

import torch

from chimera_ml.core.registry import LOSSES, MODELS, OPTIMIZERS
from common.callbacks.wsm_pseudo_scale_warmup_callback import WSMPseudoScaleWarmupCallback
from common.callbacks.wsm_r4_balance_callback import WSMR4BalanceCallback
from common.loss.wsm_masked_sparse_loss import WSMMaskedSparseLoss
from fusion.data.wsm_av_fusion_datamodule import WSMAVFusionDataModule
from fusion.data.wsm_ramps_semantic_datamodule import WSMRampsSemanticDataModule
from fusion.loss.r3_aux_agreement_loss import R3AuxAgreementLoss
from fusion.loss.r4_ramps_balance_loss import WSMR4RampsBalanceLoss
import fusion.models.av_audio_query_temporal_video  # noqa: F401
import fusion.models.av_r3_disease_query  # noqa: F401
import fusion.loss.r3_aux_agreement_loss  # noqa: F401
import fusion.loss.r4_ramps_balance_loss  # noqa: F401
import common.optimizers  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
AUDIO_CHECKPOINT = ROOT / "logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt"
PSEUDO_CACHE = Path("/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt")
COMMON = dict(
    data_root="/media/maxim/Databases/WSM_NEW",
    audio_feature_cache_root="/media/maxim/Databases/WSM_NEW/features",
    video_cache_root="/media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache",
    batch_size=2, num_workers=0, pin_memory=False, persistent_workers=False,
    shuffle_train=False, drop_last_train=False,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_b_count(hidden: int, residual: int) -> int:
    # LayerNorm/Linear/MHA/2 task-specific residual heads; frozen adapter excluded.
    return (2 * 512 + 512 * hidden + hidden) + (2 * 192 + 192 * hidden + hidden) + (4 * hidden * hidden + 4 * hidden) + 2 * (4 * hidden + 2 * hidden * residual + residual + residual + 1)


def r3_count(hidden: int, gate: int) -> int:
    audio = 2 * 768 + 768 * hidden + hidden
    video = 2 * 512 + 512 * hidden + hidden
    queries = 2 * hidden
    candidate_norms = 2 * (2 * hidden)
    gate_block = 4 * hidden + 2 * hidden * gate + gate + gate + 1
    fusion_norms = 2 * (2 * hidden)
    main = 2 * (2 * hidden + hidden * hidden + hidden + hidden + 1)
    aux = 4 * (2 * hidden + hidden + 1)
    return audio + video + queries + candidate_norms + gate_block + fusion_norms + main + aux


def assert_caps() -> None:
    b_counts = [candidate_b_count(h, r) for h, r in product((96, 128, 160, 192, 224, 256), (64, 96, 128, 160, 192, 256))]
    r_counts = [r3_count(h, g) for h, g in product((128, 160, 192, 224, 256), (96, 128, 160, 192, 256))]
    assert max(b_counts) <= 1_000_000, max(b_counts)
    assert max(r_counts) <= 736_004, max(r_counts)
    for h, heads, residual in product((96, 128, 160, 192, 224, 256), (2, 4, 8), (64, 96, 128, 160, 192, 256)):
        assert h % heads == 0
    print("PARAMETER_CAPS_PASS", max(b_counts), max(r_counts), len(b_counts), len(r_counts))


def build_model(name: str, **params: object) -> torch.nn.Module:
    model = MODELS.get(name)(**params)
    return model


def trainable_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def smoke() -> None:
    dm = WSMAVFusionDataModule(**COMMON)
    batch = next(iter(dm.train_dataloader()))
    observed = batch.masks["observed_mask"].bool()
    assert torch.isnan(batch.targets[~observed]).all()

    candidate = build_model(
        "wsm_av_audio_query_temporal_video_model",
        audio_checkpoint_path=str(AUDIO_CHECKPOINT), audio_feature_dim=768,
        audio_hidden_dim=192, video_feature_dim=512, hidden_dim=128,
        num_heads=4, residual_hidden_dim=128, dropout=0.2, num_tasks=2,
    )
    candidate.eval()
    output = candidate(batch)
    assert output.preds.shape == (2, 2) and torch.isfinite(output.preds).all()
    assert torch.equal(output.preds, output.aux["audio_base_logits"])
    loss = WSMMaskedSparseLoss()(output, batch)
    loss.backward()
    assert torch.isfinite(loss)
    optimizer = OPTIMIZERS.get("wsm_trainable_adamw_optimizer")(model=candidate, lr=1e-4, weight_decay=0.01)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    frozen_ids = {id(parameter) for parameter in candidate.audio_adapter.parameters()}
    assert not optimizer_ids & frozen_ids
    assert all(parameter.grad is None for parameter in candidate.audio_adapter.parameters())
    print("CANDIDATE_B_SMOKE_PASS", trainable_count(candidate), len(optimizer_ids))

    r3 = build_model("wsm_av_r3_disease_query_model", audio_feature_dim=768, video_feature_dim=512, hidden_dim=192, gate_hidden_dim=192, dropout=0.2, num_tasks=2)
    r3_out = r3(batch)
    r3_loss = R3AuxAgreementLoss()(r3_out, batch)
    r3_loss.backward()
    assert r3_out.preds.shape == (2, 2) and torch.isfinite(r3_out.preds).all() and torch.isfinite(r3_loss)
    print("R3B_SMOKE_PASS", trainable_count(r3))

    semantic = WSMRampsSemanticDataModule(pseudo_cache_path=str(PSEUDO_CACHE), **COMMON)
    semantic_batch = next(iter(semantic.train_dataloader()))
    sem_obs = semantic_batch.masks["observed_mask"].bool()
    accept = semantic_batch.masks["pseudo_accept_mask"].bool()
    assert not bool((sem_obs & accept).any())
    pseudo = semantic_batch.inputs["pseudo_targets"].detach().clone().requires_grad_(True)
    reliability = semantic_batch.inputs["pseudo_reliability"].detach().clone().requires_grad_(True)
    semantic_batch.inputs["pseudo_targets"] = pseudo
    semantic_batch.inputs["pseudo_reliability"] = reliability
    r4 = build_model("wsm_av_r3_disease_query_model", audio_feature_dim=768, video_feature_dim=512, hidden_dim=192, gate_hidden_dim=192, dropout=0.2, num_tasks=2)
    r4_out = r4(semantic_batch)
    r4_loss = WSMR4RampsBalanceLoss(mode="ra_stch", pseudo_scale=0.0, aux_weight=0.25, agreement_weight=0.10, tau=0.1, progress_temperature=0.25, progress_reference_depression=0.697035, progress_reference_parkinson=0.852104, controller_ema=0.8, grad_ema=0.9, reliability_ema=0.9, weight_min=0.2, weight_max=0.8, eps=1e-8)(r4_out, semantic_batch)
    r4_loss.backward()
    assert r4_out.preds.shape == (2, 2) and torch.isfinite(r4_out.preds).all() and torch.isfinite(r4_loss)
    assert pseudo.grad is None and reliability.grad is None
    assert "dev/depression/score" in inspect.getsource(WSMR4BalanceCallback.on_epoch_end)
    assert "dev/parkinson/score" in inspect.getsource(WSMR4BalanceCallback.on_epoch_end)
    assert "test_" not in inspect.getsource(WSMR4BalanceCallback.on_epoch_end)
    assert WSMPseudoScaleWarmupCallback(3, 5, 1.0).scale_for_epoch(1) == 0.0
    print("R4_SMOKE_PASS", trainable_count(r4), "PSEUDO_DETACHED", "CONTROLLER_DEV_ONLY")


def main() -> None:
    assert AUDIO_CHECKPOINT.is_file()
    assert sha256(AUDIO_CHECKPOINT) == "0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2"
    assert PSEUDO_CACHE.is_file()
    assert sha256(PSEUDO_CACHE) == "17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945"
    assert_caps()
    smoke()
    print("FROZEN_HASHES_PASS")
    print("TRIPLE_OPTUNA_FIREWALL_PASS")


if __name__ == "__main__":
    main()
