#!/usr/bin/env python3
"""TASK-003D frozen Qwen3-VL observable-description preflight."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
REQUESTED_REVISION = "1dd1e02d981403da25ed73d43430e4ef598cb94b"
PROMPT = ("Describe only directly observable visual behavior in this short video clip.\n"
          "Focus on facial movement or expression, gaze and head motion, hand or body movement,\n"
          "posture, interaction or engagement, and recording/view conditions when they are visible.\n"
          "Do not infer or mention any diagnosis, disease, health condition, neurological or psychiatric\n"
          "state, medication, cause, identity, age, sex or gender, race or ethnicity, or dataset,\n"
          "corpus, task, label, score, or prediction.\n"
          "Do not guess unobservable facts. Use concise neutral factual language.\n"
          "If an aspect is not visible, omit it.")
PROMPT_HASH = hashlib.sha256(PROMPT.encode()).hexdigest()
VIDEO_NUM_FRAMES = 8
STRATA = (("depression", "train"), ("depression", "dev"), ("parkinson", "train"), ("parkinson", "dev"))


def _package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "opencv-python", "accelerate", "safetensors", "Pillow", "huggingface-hub")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


def _stable_key(row: dict[str, Any]) -> str:
    segment_file = row.get("segment_file")
    if segment_file is None:
        segment_file = json.loads(str(row["segment_id"]))[2]
    return "|".join(str(value) for value in (row["corpus"], row["split"], row["video_id"], segment_file))


def select_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for corpus, split in STRATA:
        candidates = [row for row in rows if row["corpus"] == corpus and row["split"] == split]
        ordered = sorted(candidates, key=lambda row: (hashlib.sha256(_stable_key(row).encode()).hexdigest(), _stable_key(row)))
        result.extend(ordered[:6])
    if len(result) != 24 or any(row["split"] not in {"train", "dev"} for row in result):
        raise RuntimeError("deterministic sample must contain exactly 24 TRAIN/DEV rows")
    return result


def _assert_prompt_firewall() -> None:
    assert PROMPT == ("Describe only directly observable visual behavior in this short video clip.\n"
                      "Focus on facial movement or expression, gaze and head motion, hand or body movement,\n"
                      "posture, interaction or engagement, and recording/view conditions when they are visible.\n"
                      "Do not infer or mention any diagnosis, disease, health condition, neurological or psychiatric\n"
                      "state, medication, cause, identity, age, sex or gender, race or ethnicity, or dataset,\n"
                      "corpus, task, label, score, or prediction.\n"
                      "Do not guess unobservable facts. Use concise neutral factual language.\n"
                      "If an aspect is not visible, omit it.")
    forbidden = ("diagnosis", "disease", "corpus", "video_id", "segment_file", "label", "split")
    assert not any(f"{item}:" in PROMPT for item in forbidden)
    assert VIDEO_NUM_FRAMES == 8


def _canonical_segment_path(row: dict[str, Any], data_root: str | Path) -> Path:
    segment_file = json.loads(str(row["segment_id"]))[2]
    return Path(data_root) / str(row["corpus"]) / (str(row["split"]) + "_labels") / str(row["video_id"]) / "segments" / segment_file


def _structural_audit(rows: list[dict[str, Any]], data_root: str | Path) -> dict[str, Any]:
    import cv2
    counts: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    for row in rows:
        split_key = f"{row['corpus']}/{row['split']}"
        path = _canonical_segment_path(row, data_root)
        counts[f"{split_key}/total"] += 1
        extensions[path.suffix.lower()] += 1
        if not path.is_file():
            counts[f"{split_key}/missing"] += 1
            failures.append({"key": _stable_key(row), "kind": "missing"})
            continue
        counts[f"{split_key}/existing"] += 1
        if path.stat().st_size == 0:
            counts[f"{split_key}/zero_byte"] += 1
            failures.append({"key": _stable_key(row), "kind": "zero_byte"})
            continue
        cap = cv2.VideoCapture(str(path))
        opened = bool(cap.isOpened())
        ok, frame = cap.read() if opened else (False, None)
        cap.release()
        if not opened or not ok or frame is None:
            counts[f"{split_key}/decode_failure"] += 1
            failures.append({"key": _stable_key(row), "kind": "decode_failure"})
        else:
            counts[f"{split_key}/decode_ok"] += 1
    return {"counts": dict(sorted(counts.items())), "extensions": dict(sorted(extensions.items())), "failures": failures}


def _self_test() -> None:
    _assert_prompt_firewall()
    rows = [{"corpus": corpus, "split": split, "video_id": f"v{i}", "segment_file": f"s{i}.mp4", "segment_path": "/missing"} for corpus, split in STRATA for i in range(7)]
    a = select_sample(rows)
    b = select_sample(list(reversed(rows)))
    assert [_stable_key(x) for x in a] == [_stable_key(x) for x in b]
    assert len(a) == 24 and all(x["split"] in {"train", "dev"} for x in a)
    assert all("test" not in _stable_key(x) for x in a)
    with __import__("tempfile").TemporaryDirectory() as d:
        p = Path(d) / "report.json"
        p.write_text("old", encoding="utf-8")
        assert p.exists()
    print("SELF_TEST_OK")


def _load_model():
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REQUESTED_REVISION)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
    kwargs: dict[str, Any] = {"revision": REQUESTED_REVISION, "torch_dtype": dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    return model, processor, dtype


def _generate(model: Any, processor: Any, path: str, dtype: Any) -> tuple[str, float]:
    import torch
    messages = [{"role": "user", "content": [{"type": "video", "video": path}, {"type": "text", "text": PROMPT}]}]
    started = time.perf_counter()
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", processor_kwargs={"videos_kwargs": {"num_frames": VIDEO_NUM_FRAMES, "fps": None}})
    target_device = next(model.parameters()).device
    inputs = {key: value.to(target_device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=120, do_sample=False, num_beams=1, num_return_sequences=1)
    prompt_len = inputs["input_ids"].shape[1]
    text = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return text, time.perf_counter() - started


def _model_identity(model: Any, processor: Any, model_dir: str | None = None) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download
    index_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json", revision=REQUESTED_REVISION)
    index_sha = hashlib.sha256(Path(index_path).read_bytes()).hexdigest()
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    shard_hashes = {name: hashlib.sha256(Path(hf_hub_download(MODEL_ID, name, revision=REQUESTED_REVISION)).read_bytes()).hexdigest() for name in shard_names}
    return {"model": MODEL_ID, "requested_revision": REQUESTED_REVISION, "resolved_commit": getattr(getattr(model, "config", None), "_commit_hash", None), "processor_class": type(processor).__name__, "model_class": type(model).__name__, "safetensors_index_sha256": index_sha, "model_shard_sha256": shard_hashes}


def run(args: argparse.Namespace) -> None:
    _assert_prompt_firewall()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output}")
    from common.data.wsm_manifest import build_manifest
    rows, manifest_audit = build_manifest(args.data_root)
    structural = _structural_audit(rows, args.data_root)
    sample = select_sample(rows)
    model, processor, dtype = _load_model()
    identity = _model_identity(model, processor)
    generated = []
    for row in sample:
        source_path = _canonical_segment_path(row, args.data_root)
        text, seconds = _generate(model, processor, str(source_path), dtype)
        generated.append({"sample_key": _stable_key(row), "source_path": str(source_path), "description": text, "wall_seconds": seconds, "manual_audit": {"nonempty": bool(text), "visually_grounded": None, "major_unsupported": None, "diagnosis_or_health": None, "demographic_or_identity": None, "causal_or_medication": None, "dataset_task_label_leakage": None, "concise_for_semantic_encoding": None}})
    first = str(_canonical_segment_path(sample[0], args.data_root))
    repeat_a, repeat_seconds_a = _generate(model, processor, first, dtype)
    repeat_b, repeat_seconds_b = _generate(model, processor, first, dtype)
    generated[0]["repeat_generation"] = {"equal": repeat_a == repeat_b, "hash_a": hashlib.sha256(repeat_a.encode()).hexdigest(), "hash_b": hashlib.sha256(repeat_b.encode()).hexdigest(), "wall_seconds_a": repeat_seconds_a, "wall_seconds_b": repeat_seconds_b}
    payload = {"schema": "wsm_t2_observable_description_preflight_v1", "model": identity, "prompt": {"sha256": PROMPT_HASH, "text": PROMPT}, "generation": {"do_sample": False, "max_new_tokens": 120, "num_return_sequences": 1, "num_beams": 1, "temperature_or_top_p_tuned": False, "system_prompt": None, "video_sampling": {"num_frames": VIDEO_NUM_FRAMES, "mode": "uniform_deterministic", "reason": "single transparent CUDA-memory runtime correction after default video processing OOM"}}, "environment": {"package_versions": _package_versions(), "device": str(next(model.parameters()).device), "dtype": str(dtype), "cuda_peak_allocated": __import__("torch").cuda.max_memory_allocated() if __import__("torch").cuda.is_available() else None, "cuda_peak_reserved": __import__("torch").cuda.max_memory_reserved() if __import__("torch").cuda.is_available() else None}, "source_audit": structural, "manifest_fingerprint": manifest_audit["manifest_fingerprint"], "sample_keys": [_stable_key(row) for row in sample], "samples": generated, "test_generation": False, "test_manual_inspection": False, "no_performance_metrics": True, "training_performed": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "sample_count": len(sample), "model": identity, "test_generation": False}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen TASK-003D Qwen3-VL observable-description preflight.")
    parser.add_argument("--data-root", default="/media/maxim/Databases/WSM_NEW")
    parser.add_argument("--output", required=False, default="/media/maxim/Programs/Features/WSM/stage3_t2_preflight/qwen3vl8b_observable_preflight_v1.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
