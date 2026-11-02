"""CAREWSM registrations for the installed Chimera-ML registry."""
import importlib

_REGISTERED = False

def register() -> None:
    """Register CAREWSM components through the official Chimera-ML entrypoint."""
    global _REGISTERED
    if _REGISTERED:
        return
    modules = (
        "audio.data.chimera_datamodule",
        "audio.models.classifiers",
        "audio.loss.masked_bce",
        "audio.metrics.classification",
        "audio.callbacks.summary",
        "audio.callbacks.val_selection",
    )
    for module in modules:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name != "mamba_ssm":
                raise
    _REGISTERED = True
