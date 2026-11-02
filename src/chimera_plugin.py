"""CAREWSM registrations for the installed Chimera-ML registry."""
import importlib

_REGISTERED = False

def register() -> None:
    """Register CAREWSM components through Chimera-ML's official entrypoint."""
    global _REGISTERED
    if _REGISTERED:
        return
    modules = (
        "src.audio.data.chimera_datamodule",
        "src.audio.models.classifiers",
        "src.audio.loss.masked_bce",
        "src.audio.metrics.classification",
    )
    for module in modules:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name != "mamba_ssm":
                raise
    _REGISTERED = True
