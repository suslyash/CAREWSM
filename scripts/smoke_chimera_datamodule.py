"""Build and inspect the real joint DataModule through Chimera-ML 0.3.0."""
from chimera_ml.core.config import load_yaml
from chimera_ml.training.builders import BuildContext, build_datamodule
import chimera_plugin
chimera_plugin.register()
cfg = load_yaml("configs/audio/joint_native_fold0.yaml")
context = BuildContext(config=cfg, stage="smoke")
dm = build_datamodule(cfg["data"], context=context)
raw = {"TRAIN": dm.train_dataloader(), "VAL": dm.val_dataloader(),
       "CARE TEST": dm.test_dataloader()["CARE"], "WSM TEST": dm.test_dataloader()["WSM"]}
for name, loaders in raw.items():
    items = loaders.items() if isinstance(loaders, dict) else [(name, loaders)]
    for subname, loader in items:
        batch = next(iter(loader))
        print(name, subname if subname != name else "", "samples=", len(loader.dataset),
              "native=", tuple(batch.inputs["native"].shape),
              "sequence_mask=", tuple(batch.masks["sequence_mask"].shape),
              "meta=", len(batch.meta["sample_meta"]))
print("context=", context.values)
