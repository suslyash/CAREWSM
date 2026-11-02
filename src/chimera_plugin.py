"""CAREWSM registrations for the installed Chimera-ML registry."""
from chimera_ml.core.registry import DATAMODULES
_REGISTERED = False

def register() -> None:
    """Register CAREWSM components through Chimera-ML's official entrypoint."""
    global _REGISTERED
    if _REGISTERED:
        return
    from src.audio.data.chimera_datamodule import CarewsmJointDataModule
    from src.audio import models as _models  # noqa: F401
    if "carewsm_joint" not in DATAMODULES.keys():
        DATAMODULES.register("carewsm_joint")(CarewsmJointDataModule)
    _REGISTERED = True
