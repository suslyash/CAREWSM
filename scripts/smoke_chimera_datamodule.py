"""Build and inspect the real joint DataModule through Chimera-ML."""

from chimera_ml.core.config import load_yaml
from chimera_ml.training.builders import BuildContext, build_datamodule

import chimera_plugin


def main() -> None:
    chimera_plugin.register()
    config = load_yaml("configs/audio/joint_native_fold0.yaml")
    context = BuildContext(config=config, stage="smoke")
    data_module = build_datamodule(config["data"], context=context)
    loaders = {"train": data_module.train_dataloader()}
    loaders.update(data_module.val_dataloader())
    for name, loader in loaders.items():
        batch = next(iter(loader))
        print(
            name,
            "samples=",
            len(loader.dataset),
            "native=",
            tuple(batch.inputs["native"].shape),
            "sequence_mask=",
            tuple(batch.masks["sequence_mask"].shape),
            "meta=",
            len(batch.meta["sample_meta"]),
        )
    print("context=", context.values)


if __name__ == "__main__":
    main()
