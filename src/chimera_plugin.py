"""CAREWSM registrations for the installed Chimera-ML registry."""
import importlib

_REGISTERED = False

def register() -> None:
    """Register CAREWSM components through the official Chimera-ML entrypoint."""
    global _REGISTERED
    if _REGISTERED:
        return
    modules = (
        "audio.data.carewsm_audio_datamodule",
        "audio.models.carewsm_audio_models",
        "audio.models.carewsm_representation_pilot_models",
        "audio.loss.carewsm_masked_bce_loss",
        "audio.loss.carewsm_weighted_bce_loss",
        "audio.loss.carewsm_stageg_mtl_loss",
        "audio.metrics.carewsm_audio_metrics",
        "audio.callbacks.carewsm_summary_callback",
        "audio.callbacks.carewsm_validation_monitor_callback",
        "audio.callbacks.carewsm_stageg_mtl_callback",
    )
    for module in modules:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name != "mamba_ssm":
                raise
    _REGISTERED = True
