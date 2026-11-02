"""CAREWSM registrations for the installed Chimera-ML registry."""
import importlib

_REGISTERED = False

def register() -> None:
    """Register CAREWSM components through Chimera-ML's official entrypoint."""
    global _REGISTERED
    if _REGISTERED:
        return
    modules = (
        "audio.data.chimera_datamodule",
        "audio.models.classifiers",
        "audio.loss.masked_bce",
        "audio.metrics.classification",
        "audio.callbacks.summary",
    )
    for module in modules:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name != "mamba_ssm":
                raise
    _REGISTERED = True
