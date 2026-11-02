"""Small real-data TASK-002 smoke; no training or feature rescan."""
from __future__ import annotations
import argparse
from pathlib import Path
from src.audio.data.joint_datamodule import JointDataModule

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--care-root", type=Path, default=Path("/media/maxim/Databases/CARE")); p.add_argument("--wsm-root", type=Path, default=Path("/media/maxim/Databases/WSM_NEW")); p.add_argument("--manifest", type=Path, default=Path("reports/feature_audit/sample_feature_availability.csv")); p.add_argument("--registry", type=Path, default=Path("/media/maxim/Databases/CARE/splits/features.json")); a = p.parse_args()
    dm = JointDataModule(a.care_root, a.wsm_root, a.manifest, a.registry)
    for k in range(5):
        views = dm.fold(k, ("wav2vec_xlsr300m",))
        print(k, *(f"{n}={len(v)}/{v.profiles}" for n, v in views.items()), "labels", views["train"].label_counts())
    both = dm.fold(0, ("egemaps_func", "wav2vec_xlsr300m"))
    print("two_family", len(both["train"]), len(both["val"]), len(both["care_test"]), len(both["wsm_test"]))
    sample = next(iter(dm.fold(0)["train"])); feature = dm.load_native(sample, "egemaps_func")
    embedding = dm.load_native(sample, "emotionmsp_embeds")
    print("representative", sample.key, feature.values.shape, sorted(feature.metadata), feature.values.dtype)
    print("npz_embed", embedding.values.shape, embedding.values.dtype)

if __name__ == "__main__": main()
