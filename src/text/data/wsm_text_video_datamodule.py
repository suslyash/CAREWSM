from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset
from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import DATAMODULES
from chimera_ml.data.datamodule import DataModule
from common.data.wsm_manifest import build_manifest


TASKS = ("depression", "parkinson")


def _load_index(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    return {(x["corpus"], x["split"], x["video_id"]): x for x in entries}


class _TextDataset(Dataset):
    def __init__(self, samples: Iterable[dict[str, Any]], cache_root: Path) -> None:
        self.samples = list(samples)
        self.cache_root = cache_root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        artifact = torch.load(self.cache_root / sample["artifact_path"], map_location="cpu", weights_only=False)
        features = artifact["features"].to(dtype=torch.float32)
        return {
            "features": features,
            "targets": torch.tensor(sample["targets"], dtype=torch.float32),
            "observed": torch.tensor(sample["observed"], dtype=torch.bool),
            "meta": sample["meta"],
        }


def collate_wsm_text_t1(samples: list[dict[str, Any]]) -> Batch:
    if not samples:
        raise ValueError("cannot collate empty T1 batch")
    max_chunks = max(int(item["features"].shape[0]) for item in samples)
    values = torch.zeros(len(samples), max_chunks, 768, dtype=torch.float32)
    mask = torch.zeros(len(samples), max_chunks, dtype=torch.bool)
    for i, item in enumerate(samples):
        count = item["features"].shape[0]
        values[i, :count] = item["features"]
        mask[i, :count] = True
    return Batch(
        inputs={"text": values},
        targets=torch.stack([item["targets"] for item in samples]),
        masks={"text_mask": mask, "observed_mask": torch.stack([item["observed"] for item in samples])},
        meta={"sample_meta": [item["meta"] for item in samples]},
    )


def _filter_value(value: Any) -> bool:
    return str(value).strip() in {"1", "1.0", "True", "true"}


def _filter_map(root: Path) -> dict[tuple[str, str, str], dict[str, bool]]:
    result: dict[tuple[str, str, str, str], dict[str, bool]] = {}
    for corpus in TASKS:
        for split in ("train", "dev", "test"):
            suffix = "_mishas" if split == "test" else ""
            path = root / corpus / f"{split}_labels_segments_min_filtered{suffix}.csv"
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle, delimiter=";" if suffix else ","):
                    result[(corpus, split, row["video_id"], row["segment_file"])] = {"soft": _filter_value(row.get("soft_filter")), "hard": _filter_value(row.get("hard_filter"))}
    return result


@dataclass
class WSMTextT1DataModule(DataModule):
    data_root: str = "/media/maxim/Databases/WSM_NEW"
    text_cache_root: str = "/media/maxim/Programs/Features/WSM/text_t1_xlmr_v1"
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    shuffle_train: bool = True
    drop_last_train: bool = False

    def __post_init__(self) -> None:
        rows, _ = build_manifest(self.data_root)
        root = Path(self.text_cache_root).expanduser().resolve()
        index_path = root / "cache_index.json"
        index = _load_index(index_path)
        self.cache_index_sha256 = hashlib.sha256(index_path.read_bytes()).hexdigest()
        self.cache_fingerprint = json.loads(index_path.read_text(encoding="utf-8")).get("cache_fingerprint", "")
        filters = _filter_map(Path(self.data_root))
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in rows:
            key = (row["corpus"], row["split"], row["video_id"])
            grouped.setdefault(key, []).append(row)

        def make_sample(key: tuple[str, str, str], row_group: list[dict[str, Any]], *, segment: dict[str, Any] | None = None) -> dict[str, Any] | None:
            entry = index[key]
            if not entry["available"]:
                return None
            for row in row_group:
                if row["corpus"] == "depression":
                    assert row["y_depression"] == row_group[0]["y_depression"]
                else:
                    assert row["y_parkinson"] == row_group[0]["y_parkinson"]
            first = row_group[0]
            targets = [float(first["y_depression"]) if first["y_depression"] is not None else float("nan"), float(first["y_parkinson"]) if first["y_parkinson"] is not None else float("nan")]
            observed = [first["observed_depression"], first["observed_parkinson"]]
            chosen = segment or first
            return {"artifact_path": entry["artifact_path"], "targets": targets, "observed": observed, "meta": {"corpus": key[0], "split": key[1], "video_id": key[2], "segment_id": chosen["segment_id"], "unit": "video" if segment is None else "segment"}}

        train = [sample for key, group in grouped.items() if key[1] == "train" for sample in [make_sample(key, group)] if sample is not None]
        self.train_dataset = _TextDataset(train, root)
        self.train_unit_counts = {corpus: sum(item["meta"]["corpus"] == corpus for item in train) for corpus in TASKS}
        eval_sets: dict[str, Dataset] = {}
        dev = [sample for key, group in grouped.items() if key[1] == "dev" for row in group for sample in [make_sample(key, group, segment=row)] if sample is not None]
        eval_sets["dev"] = _TextDataset(dev, root)
        for name in ("none", "soft", "hard"):
            selected = []
            for key, group in grouped.items():
                if key[1] != "test":
                    continue
                for row in group:
                    segment_file = json.loads(row["segment_id"])[2]
                    if name == "none" or filters.get((key[0], key[1], key[2], segment_file), {}).get(name, False):
                        sample = make_sample(key, group, segment=row)
                        if sample is not None:
                            selected.append(sample)
            eval_sets[f"test_{name}"] = _TextDataset(selected, root)
        self.val_dataset = eval_sets
        self.test_dataset = {key: value for key, value in eval_sets.items() if key.startswith("test_")}
        self.collate_fn = collate_wsm_text_t1
        self.feature_dim = 768

    def describe_context(self, context: Any) -> None:
        context.set("data.feature_dim", 768)
        context.set("data.text_feature_dim", 768)
        context.set("data.num_tasks", 2)
        context.set("data.task_names", list(TASKS))
        context.set("data.class_names", list(TASKS))
        context.set("data.cache_index_sha256", self.cache_index_sha256)
        context.set("data.train_unit_semantics", "one available video-level transcript per canonical corpus/video")
        context.set("data.evaluation_unit_semantics", "canonical segment row broadcasts parent video transcript feature sequence")


@DATAMODULES.register("wsm_text_t1_datamodule")
def wsm_text_t1_datamodule(context: Any | None = None, **params: Any) -> WSMTextT1DataModule:
    return WSMTextT1DataModule(**params)
