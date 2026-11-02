"""CAREWSM registrations for the installed Chimera-ML registry."""
import importlib


def _install_flat_audio_paths() -> None:
    """Keep CAREWSM logger/callback artifacts in the configured run directory."""
    import math

    from chimera_ml.callbacks.checkpoint_callback import CheckpointCallback
    from chimera_ml.callbacks.early_stopping_callback import EarlyStoppingCallback
    from chimera_ml.callbacks.snapshot_callback import SnapshotCallback
    from chimera_ml.logging.console_file_logger import ConsoleFileLogger

    def patch_post_init(cls: type) -> None:
        original = cls.__post_init__
        if getattr(cls, "_carewsm_flat_paths", False):
            return

        def post_init(self: object) -> None:
            if str(getattr(self, "experiment_name", "")).startswith("audio_"):
                setattr(self, "run_name", "")
            original(self)

        cls.__post_init__ = post_init  # type: ignore[method-assign]
        cls._carewsm_flat_paths = True

    def patch_fit_start(cls: type) -> None:
        original = cls.on_fit_start
        if getattr(cls, "_carewsm_flat_paths", False):
            return

        def on_fit_start(self: object, trainer: object) -> None:
            if str(getattr(self, "experiment_name", "")).startswith("audio_"):
                setattr(self, "run_name", "")
            original(self, trainer)

        cls.on_fit_start = on_fit_start  # type: ignore[method-assign]
        cls._carewsm_flat_paths = True

    patch_post_init(CheckpointCallback)
    patch_fit_start(SnapshotCallback)
    patch_post_init(ConsoleFileLogger)

    checkpoint_end = CheckpointCallback.on_epoch_end
    early_end = EarlyStoppingCallback.on_epoch_end

    def checkpoint_guard(self: object, trainer: object, epoch: int, logs: dict[str, float]) -> None:
        monitor = str(getattr(self, "monitor", ""))
        if (str(getattr(self, "experiment_name", "")).startswith("audio_")
                and monitor in logs and not math.isfinite(float(logs[monitor]))):
            if getattr(self, "save_last", False):
                self._save(trainer, epoch=epoch, step=trainer.global_step,
                           monitor_value=float("nan"), is_last=True)
            trainer.logger.warning("Skipping non-finite VAL checkpoint monitor %s", monitor)
            return
        checkpoint_end(self, trainer, epoch, logs)

    def early_guard(self: object, trainer: object, epoch: int, logs: dict[str, float]) -> None:
        monitor = str(getattr(self, "monitor", ""))
        if monitor in logs and not math.isfinite(float(logs[monitor])):
            trainer.logger.warning("Skipping non-finite VAL early-stopping monitor %s", monitor)
            return
        early_end(self, trainer, epoch, logs)

    CheckpointCallback.on_epoch_end = checkpoint_guard  # type: ignore[method-assign]
    EarlyStoppingCallback.on_epoch_end = early_guard  # type: ignore[method-assign]


_install_flat_audio_paths()

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
