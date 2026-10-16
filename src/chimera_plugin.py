from __future__ import annotations

import importlib
import warnings


_MODULES_TO_REGISTER: tuple[str, ...] = (
    "common.callbacks.wsm_segment_callback",
    "common.callbacks.wsm_summary_callback",
    "common.callbacks.wsm_pseudo_scale_warmup_callback",
    "common.callbacks.wsm_r4_balance_callback",
    "common.callbacks.wsm_gradient_mtl_callback",
    "common.loss.wsm_masked_sparse_loss",
    "common.optimizers",
    "audio.data.wsm_audio_segment_datamodule",
    "audio.models.audio_mamba_segment",
    "audio.loss.wsm_audio_loss",
    "fusion.models.av_sync_mamba_segment",
    "fusion.models.av_f0_gated_late",
    "fusion.models.av_f1_shared_mtl",
    "fusion.models.av_f2_task_aware_directed",
    "fusion.models.av_f1_temporal_audio_residual",
    "fusion.models.av_audio_first_zero_residual",
    "fusion.models.av_audio_query_temporal_video",
    "fusion.models.av_audio_confidence_gated",
    "fusion.models.av_r3_disease_query",
    "fusion.data.wsm_manifest_datamodule",
    "fusion.data.wsm_av_fusion_datamodule",
    "fusion.data.wsm_ramps_semantic_datamodule",
    "fusion.loss.ramps_observed_pseudo_loss",
    "fusion.loss.r3_aux_agreement_loss",
    "fusion.loss.r4_ramps_balance_loss",
    "fusion.loss.gradient_mtl_loss",
    "video.models.depart_v1",
    "video.models.depart_v2",
    "video.data.wsm_video_cache_datamodule",
    "text.features.xlmr_video_transcript",
    "text.data.wsm_text_video_datamodule",
    "text.models.t1_chunk_transformer",
)


def register() -> None:
    for module_name in _MODULES_TO_REGISTER:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name in {"mamba_ssm", "torch", "pandas", "numpy", "matplotlib"}:
                warnings.warn(f"Optional dependency is missing while importing '{module_name}': {exc}", stacklevel=2)
                continue

            warnings.warn(f"Failed to import '{module_name}': {exc}", stacklevel=2)
        except Exception as exc:
            warnings.warn(f"Failed to import '{module_name}': {exc}", stacklevel=2)
