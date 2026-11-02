"""Leak-safe, native-feature data layer for the joint CARE+WSM protocol."""
from __future__ import annotations

import csv, hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

FAMILIES = ("egemaps_func", "egemaps_lld", "disvoice_prosody_static", "disvoice_prosody_dynamic",
            "disvoice_phonation_static", "disvoice_phonation_dynamic", "disvoice_articulation_static",
            "wav2vec_xlsr300m", "trillsson4_1", "emotionmsp_trajectories", "emotionmsp_embeds")

class DataContractError(ValueError): pass

@dataclass(frozen=True)
class Sample:
    corpus: str; sample_id: str; profile_id: str; video_id: str; subgroup: str
    label_id: str; root: Path; feature_dir: Path
    depression: float | None; parkinson: float | None
    depression_observed: bool; parkinson_observed: bool
    provenance: tuple[str, ...]
    availability: dict[str, str]

    @property
    def key(self) -> tuple[str, str]: return (self.corpus, self.sample_id)

@dataclass(frozen=True)
class View:
    name: str; fold: int | None; rows: tuple[Sample, ...]; fingerprint: str
    exclusions: dict[str, int]

    def __iter__(self) -> Iterator[Sample]: return iter(self.rows)
    def __len__(self) -> int: return len(self.rows)
    @property
    def profiles(self) -> int: return len({(r.corpus, r.profile_id) for r in self.rows})
    def label_counts(self) -> dict[str, int]:
        return {t: sum(getattr(r, t + "_observed") for r in self.rows) for t in ("depression", "parkinson")}

@dataclass(frozen=True)
class NativeFeature:
    values: np.ndarray; metadata: dict[str, np.ndarray]; family: str; path: Path
    def __post_init__(self):
        if self.values.ndim != 2 or self.values.shape[0] == 0 or not np.isfinite(self.values).all():
            raise DataContractError(f"invalid numeric feature array for {self.path}: {self.values.shape}")

def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))

def _float_label(value: str) -> float | None:
    if value is None or str(value).strip() == "": return None
    try:
        x = float(value)
        if x not in (0.0, 1.0): raise ValueError
        return x
    except ValueError as e: raise DataContractError(f"label must be binary or missing: {value!r}") from e

class JointDataModule:
    def __init__(self, care_root: str | Path, wsm_root: str | Path,
                 manifest: str | Path, registry: str | Path | None = None):
        self.roots = {"CARE": Path(care_root), "WSM": Path(wsm_root)}
        self.manifest_path = Path(manifest)
        self.registry = json.loads(Path(registry).read_text()) if registry else None
        manifest_rows = _rows(self.manifest_path)
        self._manifest = {}
        for r in manifest_rows:
            key = (r["corpus"], r["sample_id"])
            if key in self._manifest: raise DataContractError(f"duplicate manifest key: {key}")
            self._manifest[key] = r
        self._source = {c: self._read_sources(c) for c in self.roots}
        self._samples = {k: self._make_sample(k, r) for k, r in self._manifest.items()}
        self._validate_labels()

    def _read_sources(self, corpus: str) -> dict[str, dict[str, dict[str, str]]]:
        root = self.roots[corpus]; split = root / "splits"
        out = {}
        for name in ("train", "dev", "test"):
            out[name] = {r["sample_id"]: r for r in _rows(split / f"{name}.csv")}
        return out

    def _make_sample(self, key, m):
        corpus, sid = key; src = next((self._source[corpus][n].get(sid) for n in ("dev", "train", "test") if sid in self._source[corpus][n]), None)
        if src is None: raise DataContractError(f"manifest key absent from source splits: {key}")
        profile, video = src["profile_id"], src["video_id"]
        profile_dir = profile if profile.startswith("PROFILE_") else "PROFILE_" + profile
        video_dir = video if video.startswith("VIDEO_") else "VIDEO_" + video
        label = src.get("label_id", "")
        feature_dir = self.roots[corpus] / "extracted" / label / src.get("subgroup", "") / profile_dir / video_dir
        availability = {f: m.get(f + "_status", "UNKNOWN") for f in FAMILIES}
        return Sample(corpus, sid, profile, video, src.get("subgroup", ""), label, self.roots[corpus], feature_dir,
                      _float_label(m.get("depression", "")), _float_label(m.get("parkinson", "")),
                      m.get("depression_observed", "False") == "True", m.get("parkinson_observed", "False") == "True",
                      tuple(sorted(set(m.get("source_declared_files", "").split(";")) - {""})), availability)

    def _validate_labels(self):
        for s in self._samples.values():
            for t in ("depression", "parkinson"):
                if getattr(s, t + "_observed") != (getattr(s, t) is not None):
                    raise DataContractError(f"label observed mismatch: {s.key} {t}")

    def _view(self, name: str, rows: Iterable[Sample], fold: int | None = None, selected: tuple[str, ...] = ()) -> View:
        exclusions = {"missing_source": 0, "feature_unavailable": 0}; kept = []
        for s in rows:
            if any(s.availability.get(f) != "VALID" for f in selected): exclusions["feature_unavailable"] += 1
            else: kept.append(s)
        kept.sort(key=lambda s: s.key)
        payload = "\n".join(f"{s.corpus}|{s.sample_id}" for s in kept).encode()
        return View(name, fold, tuple(kept), hashlib.sha256(payload).hexdigest(), exclusions)

    def _fold(self, corpus: str, fold: int, part: str) -> list[Sample]:
        path = self.roots[corpus] / "splits" / "folds" / f"fold_{fold}_{part}.csv"
        source = _rows(path); seen = set(); out = []
        for r in source:
            key = (corpus, r["sample_id"])
            if key in seen: raise DataContractError(f"duplicate fold key: {key}")
            seen.add(key)
            if key not in self._samples: raise DataContractError(f"fold key absent from manifest: {key}")
            if r.get("profile_id") != self._samples[key].profile_id: raise DataContractError(f"profile mismatch: {key}")
            out.append(self._samples[key])
        return out

    def fold(self, fold: int, selected_features: Iterable[str] = ()) -> dict[str, View]:
        if fold not in range(5): raise ValueError("fold must be 0..4")
        selected = tuple(selected_features)
        unknown = set(selected) - set(FAMILIES)
        if unknown: raise ValueError(f"unknown feature family: {sorted(unknown)}")
        train = sum((self._fold(c, fold, "train") for c in self.roots), [])
        val = sum((self._fold(c, fold, "val") for c in self.roots), [])
        self._assert_isolated(train, val, "TRAIN", "VAL")
        return {"train": self._view(f"fold_{fold}_train", train, fold, selected),
                "val": self._view(f"fold_{fold}_val", val, fold, selected),
                "care_val": self._view(f"fold_{fold}_care_val", [x for x in val if x.corpus == "CARE"], fold, selected),
                "wsm_val": self._view(f"fold_{fold}_wsm_val", [x for x in val if x.corpus == "WSM"], fold, selected),
                "care_test": self.test("CARE", selected), "wsm_test": self.test("WSM", selected)}

    def test(self, corpus: str, selected_features: Iterable[str] = ()) -> View:
        rows = [self._samples[(corpus, sid)] for sid in sorted(self._source[corpus]["test"])]
        return self._view(f"{corpus.lower()}_test", rows, None, tuple(selected_features))

    def _assert_isolated(self, a, b, an, bn):
        if {x.key for x in a} & {x.key for x in b}: raise DataContractError(f"{an}/{bn} sample overlap")
        if {(x.corpus, x.profile_id) for x in a} & {(x.corpus, x.profile_id) for x in b}: raise DataContractError(f"{an}/{bn} profile overlap")

    def load_native(self, sample: Sample, family: str) -> NativeFeature:
        if family not in FAMILIES: raise ValueError(family)
        if sample.availability[family] != "VALID": raise DataContractError(f"feature is not VALID: {sample.key} {family}")
        spec = (self.registry or {}).get("sources", {}).get(family, {})
        path = sample.feature_dir / spec.get("file", family)
        if spec.get("format") == "csv":
            rows = _rows(path); metadata = set(spec.get("metadata", [])); cols = [c for c in rows[0] if c not in metadata]
            try: values = np.asarray([[float(r[c]) for c in cols] for r in rows], dtype=np.float32)
            except (KeyError, ValueError) as e: raise DataContractError(f"invalid CSV schema: {path}") from e
            return NativeFeature(values, {c: np.asarray([float(r[c]) for r in rows], dtype=np.float32) for c in metadata}, family, path)
        with np.load(path, allow_pickle=False) as z:
            arrays = [np.asarray(z[k]) for k in z.files if k not in set(spec.get("metadata", []))]
            if not arrays: raise DataContractError(f"no numeric arrays: {path}")
            values = arrays[0] if arrays[0].ndim == 2 else arrays[0].reshape(len(arrays[0]), -1)
            metadata = {k: np.asarray(z[k]) for k in z.files if k in set(spec.get("metadata", []))}
        return NativeFeature(values.astype(np.float32), metadata, family, path)

    @staticmethod
    def collate(features: Iterable[NativeFeature]) -> dict[str, Any]:
        items = list(features)
        if not items: return {"values": np.empty((0, 0, 0), np.float32), "mask": np.empty((0, 0), bool), "lengths": np.empty(0, int)}
        dim = {x.values.shape[1] for x in items}
        if len(dim) != 1: raise DataContractError("collate requires equal feature dimension")
        lengths = np.asarray([x.values.shape[0] for x in items], int); out = np.zeros((len(items), max(lengths), next(iter(dim))), np.float32)
        mask = np.zeros(out.shape[:2], bool)
        for i, x in enumerate(items): out[i, :len(x.values)] = x.values; mask[i, :len(x.values)] = True
        return {"values": out, "mask": mask, "lengths": lengths}
