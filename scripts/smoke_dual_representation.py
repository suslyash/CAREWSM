"""Tiny real-data TASK-003 smoke; no training and no Mantis cache generation."""
from pathlib import Path
import json
from audio.data.joint_datamodule import JointDataModule
from audio.features.representations import mantis_windows, native_representation, official_mantis_probe

CARE = Path("/media/maxim/Databases/CARE")
WSM = Path("/media/maxim/Databases/WSM_NEW")
MANIFEST = Path("reports/feature_audit/sample_feature_availability.csv")
REGISTRY = CARE / "splits/features.json"

dm = JointDataModule(CARE, WSM, MANIFEST, REGISTRY)
views = dm.fold(0, ["egemaps_lld"])
keys = {s.key for s in views["train"]}
assert keys == {s.key for s in views["train"]}, "paired membership changed"
for corpus in ("CARE", "WSM"):
    sample = next(s for s in views["train"] if s.corpus == corpus)
    native = native_representation(dm, sample, "egemaps_lld")
    windows = mantis_windows(dm.load_native(sample, "egemaps_lld"))
    assert native.metadata["sample_id"] == sample.sample_id
    assert windows.values.shape[1] == len(windows.metadata["windows"])
    assert windows.metadata["windows"][0][0] < windows.metadata["windows"][-1][1]
    print(json.dumps({"corpus": corpus, "sample_id": sample.sample_id,
                      "native_shape": list(native.values.shape),
                      "window_shape": list(windows.values.shape),
                      "first_window": list(windows.metadata["windows"][0]),
                      "last_window": list(windows.metadata["windows"][-1])}))
print(json.dumps({"mantis_probe": official_mantis_probe()}))
