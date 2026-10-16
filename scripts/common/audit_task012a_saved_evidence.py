#!/usr/bin/env python3
"""Strict offline validator and aggregate audit for TASK-012A evidence.

The CLI reads JSON/Markdown and hashes files only.  It does not import or run
the production evaluator, model, DataModule, callback, or inference path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any


SEEDS = (42, 43, 44, 45, 46)
PROTOCOLS = ("test_none", "test_soft", "test_hard")
TASKS = ("depression", "parkinson")
PROTOCOL_COUNTS = {"test_none": 1364, "test_soft": 1208, "test_hard": 1014}
OBSERVED_COUNTS = {
    "dev": {"depression": 621, "parkinson": 312},
    "test_none": {"depression": 827, "parkinson": 537},
    "test_soft": {"depression": 710, "parkinson": 498},
    "test_hard": {"depression": 654, "parkinson": 360},
}
DEV_COUNT = 933
T_MULTIPLIER = 2.7764451051977987
TOL = 1e-12
EXPECTED_ARTIFACTS = {
    "dev": (11291, "67a7c70754070dd0594262ab706aac8d3d853d9b2a84f349adf97e08e8ac2e64"),
    "test": (12875, "ddcae2835c04a346c1a27d3e9fa9d7c2b2afc96980b46d206dd2e79120baf285"),
    "marker": (48, "3a37730e932f29c5928b930263d6c745c6c9f9171b9c42b34990320de329e12d"),
}
FROZEN_LEDGER = {
    42: {"run_name": "task011a-shared-progress-8a53-029", "mlflow_run_id": "90a5adebf43c46808814f29374e092d0", "epoch": 7, "config_sha256": "a5bcb56f48b223f6156281183b72c8b53200cff320b3895c4cb3247fd86878d6", "checkpoint_sha256": "ba02ea341800d8d6330ecca32e8a31dde3afaf61e5fb1e1eca4fd144d1ff0346", "expected": (0.749256, 0.918143, 0.8336994328)},
    43: {"run_name": "task011b_shared_progress_parent_seed43", "mlflow_run_id": "009b4d3c9be44220b6d3169989c518e4", "epoch": 9, "config_sha256": "dda16afd9478f935107a00ddf901956b3318eb37d8385ce169e2b2e2dbb25a52", "checkpoint_sha256": "ba280153b2a1dca9808f25c05e581fec72ab3404a3602233154ef26c9f62b926", "expected": (0.7103060305, 0.7997336342, 0.7550198324)},
    44: {"run_name": "task011b_shared_progress_parent_seed44", "mlflow_run_id": "28b2b383cc244c8b98011c60c73ac14e", "epoch": 5, "config_sha256": "44c7cad4879a3635c7eff6d9a1f9a1a5cbbdbaf123ba4d19f3f9a3f2effe775b", "checkpoint_sha256": "775826bf903fced08e24eb9bfff267e414c1ea7d5e94201a466bc40642e93100", "expected": (0.7506546698, 0.8807426876, 0.8156986787)},
    45: {"run_name": "task011b_shared_progress_parent_seed45", "mlflow_run_id": "ff2b0e3033ef4688ab46841585e49bcf", "epoch": 19, "config_sha256": "f411e86536681dc69903d01d23961714e9045df9d92e431281d688c8f0145a88", "checkpoint_sha256": "e4a6776902ca57885ceb8f1c5444736d16fb8257618181c027b2fc8ad996e3d9", "expected": (0.7303546019, 0.8793876700, 0.8048711360)},
    46: {"run_name": "task011b_shared_progress_parent_seed46", "mlflow_run_id": "aa76a8669b09431ba9c6fbe3f4f98b0c", "epoch": 3, "config_sha256": "bb3db27f3ce216403cb5390ba235a8f8a0fa8f410bb6424891b5187cdcf56c60", "checkpoint_sha256": "374efc91f7ecf153f85ce42bee6af92475a229795d1221e6c8dfa4c0bb267478", "expected": (0.7407686022, 0.8589095418, 0.7998390720)},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise ValueError(message)


def finite_unit(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        fail(f"{label} must be finite and in [0,1]")
    return float(value)


def integer_count(value: Any, label: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (maximum is not None and value > maximum):
        fail(f"{label} must be a non-negative integer within membership")
    return value


def close(actual: float, expected: float, label: str, tolerance: float = TOL) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        fail(f"{label}: {actual!r} != {expected!r}")


def validate_artifact_identity(path: Path, kind: str) -> None:
    expected_bytes, expected_sha = EXPECTED_ARTIFACTS[kind]
    if path.stat().st_size != expected_bytes or sha256(path) != expected_sha:
        fail(f"{kind} artifact identity mismatch: bytes={path.stat().st_size}, sha256={sha256(path)}")


def validate_metric_block(metrics: dict[str, Any], label: str, protocol_count: int, observed: dict[str, int]) -> None:
    if set(metrics) != {"depression", "parkinson", "mean", "observed_counts", "protocol_count"}:
        fail(f"{label}: unexpected metric keys")
    if metrics["protocol_count"] != protocol_count:
        fail(f"{label}: protocol count mismatch")
    for task in TASKS:
        block = metrics[task]
        if set(block) != {"uar", "mf1", "score"}:
            fail(f"{label}/{task}: unexpected metric keys")
        uar, mf1, score = (finite_unit(block[key], f"{label}/{task}/{key}") for key in ("uar", "mf1", "score"))
        close(score, (uar + mf1) / 2.0, f"{label}/{task}/score")
        count = integer_count(metrics["observed_counts"][task], f"{label}/{task}/num_samples", protocol_count)
        if count != observed[task]:
            fail(f"{label}/{task}: observed count mismatch")
    mean_value = finite_unit(metrics["mean"], f"{label}/mean")
    close(mean_value, (metrics["depression"]["score"] + metrics["parkinson"]["score"]) / 2.0, f"{label}/mean")


def validate_frozen_files(repo: Path, entry: dict[str, Any]) -> None:
    for key in ("config", "checkpoint"):
        path = repo / entry[key]
        if not path.is_file():
            fail(f"missing frozen {key} for seed {entry['seed']}: {path}")
        expected = entry[f"{key}_sha256"]
        actual = sha256(path)
        if actual != expected or entry.get(f"{key}_sha256_actual") != expected:
            fail(f"{key} identity mismatch for seed {entry['seed']}")


def validate_dev(dev: dict[str, Any], repo: Path) -> None:
    if dev.get("schema") != "task012a-dev-preflight-v1" or dev.get("test_iteration") is not False:
        fail("DEV schema/test_iteration mismatch")
    if len(dev.get("ledger", [])) != 5 or len(dev.get("results", [])) != 5:
        fail("DEV ledger/results must each contain exactly five rows")
    if [e.get("seed") for e in dev["ledger"]] != list(SEEDS) or {r.get("seed") for r in dev["results"]} != set(SEEDS):
        fail("DEV seeds must be exactly 42..46 and unique")
    if dev.get("invariants") != {"accepted_pseudo_counts": [376, 1801], "model": "wsm_av_r3_disease_query_model", "progress_balancing": True, "pseudo_cache_sha256": "17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945", "task_aware_fusion": False, "trainable_parameters": 552775}:
        fail("DEV invariant identity mismatch")
    expected_protocol_metadata = {p: {"count": n, "expected_count": n} for p, n in PROTOCOL_COUNTS.items()}
    if dev.get("protocol_metadata") != expected_protocol_metadata:
        fail("DEV protocol metadata mismatch")
    for entry, result in zip(dev["ledger"], sorted(dev["results"], key=lambda row: row["seed"])):
        seed = entry["seed"]
        frozen = FROZEN_LEDGER[seed]
        for key in ("run_name", "mlflow_run_id", "epoch", "config_sha256", "checkpoint_sha256"):
            if entry.get(key) != frozen[key]:
                fail(f"DEV identity mismatch seed {seed}/{key}")
        for key in ("run_name", "mlflow_run_id", "epoch"):
            if result.get(key) != frozen[key]:
                fail(f"DEV result identity mismatch seed {seed}/{key}")
        if entry.get("seed") != result.get("seed") or entry.get("expected") != list(frozen["expected"]):
            fail(f"DEV ledger/result expected identity mismatch seed {seed}")
        validate_frozen_files(repo, entry)
        expected = frozen["expected"]
        metrics = result["metrics"]
        validate_metric_block(metrics, f"DEV/{seed}", DEV_COUNT, OBSERVED_COUNTS["dev"])
        actual = (metrics["depression"]["score"], metrics["parkinson"]["score"], metrics["mean"])
        for index, value in enumerate(actual):
            close(value, expected[index], f"DEV/{seed}/expected[{index}]", tolerance=0.0005)
            close(result["reproduction_difference"][index], value - expected[index], f"DEV/{seed}/reproduction_difference[{index}]")
        if result.get("model_dtype") != "torch.float32" or result.get("output_accumulation_dtype") != "torch.float32" or not isinstance(result.get("autocast_enabled"), bool) or result.get("device") not in {"cuda", "cpu"}:
            fail(f"DEV/{seed}: retained runtime metadata invalid")


def validate_test(test: dict[str, Any], dev: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    if test.get("schema") != "task012a-test-v1" or test.get("test_iteration") is not True or test.get("invocation_count") != 15:
        fail("Test schema/iteration/invocation mismatch")
    if len(test.get("results", [])) != 15:
        fail("Test results must contain exactly 15 rows before indexing")
    if test.get("protocol_counts") != PROTOCOL_COUNTS:
        fail("Test protocol counts mismatch")
    required_flags = ("no_labels", "no_raw_logits", "no_raw_predictions", "no_raw_probabilities", "no_sample_metadata")
    if any(test.get(flag) is not True for flag in required_flags):
        fail("Test no-raw-evidence declarations are incomplete")
    dev_rows = {r["seed"]: r for r in dev["results"]}
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for row in test["results"]:
        seed, protocol = row.get("seed"), row.get("protocol")
        key = (seed, protocol)
        if seed not in SEEDS or protocol not in PROTOCOLS or key in rows:
            fail(f"Test duplicate/invalid identity: {key}")
        frozen = FROZEN_LEDGER[seed]
        for field in ("run_name", "mlflow_run_id", "epoch"):
            if row.get(field) != frozen[field] or row.get(field) != dev_rows[seed].get(field):
                fail(f"Test identity mismatch {key}/{field}")
        validate_metric_block(row["metrics"], f"Test/{seed}/{protocol}", PROTOCOL_COUNTS[protocol], OBSERVED_COUNTS[protocol])
        if row.get("model_dtype") != "torch.float32" or row.get("output_accumulation_dtype") != "torch.float32" or not isinstance(row.get("autocast_enabled"), bool) or row.get("device") not in {"cuda", "cpu"}:
            fail(f"Test/{key}: retained runtime metadata invalid")
        rows[key] = row
    expected_keys = {(seed, protocol) for seed in SEEDS for protocol in PROTOCOLS}
    if set(rows) != expected_keys:
        fail(f"Test rows are not the exact seed/protocol Cartesian set: {sorted(set(rows) ^ expected_keys)}")
    return rows


def oracle_metrics(scores: list[float], labels: list[int], threshold: float = 0.0) -> dict[str, float | int]:
    if len(scores) != len(labels) or not labels:
        raise ValueError("scores and labels must be non-empty and aligned")
    prediction = [int(score >= threshold) for score in scores]
    tp = sum(p == 1 and y == 1 for p, y in zip(prediction, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(prediction, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(prediction, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(prediction, labels))
    recalls = [tn / (tn + fp) if tn + fp else None, tp / (tp + fn) if tp + fn else None]
    f1s = [2 * tn / (2 * tn + fn + fp) if 2 * tn + fn + fp else None, 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None]
    recalls = [x for x in recalls if x is not None]
    f1s = [x for x in f1s if x is not None]
    if not recalls or not f1s:
        raise ValueError("empty metric term")
    uar, mf1 = mean(recalls), mean(f1s)
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "num_samples": len(labels), "uar": uar, "mf1": mf1, "score": (uar + mf1) / 2}


def explicit_metric_oracle() -> dict[str, Any]:
    cases = [
        ([-1.0, 0.0, 1e-12, 1.0], [0, 0, 1, 1]),
        ([-3.0, 2.0, 1.0, -1.0, 0.0], [0, 1, 1, 0, 1]),
        ([-1.0, -1.0], [0, 0]),
        ([-1.0, 1.0], [0, 0]),
    ]
    outputs = [oracle_metrics(scores, labels) for scores, labels in cases]
    return {"cases": len(outputs), "all_negative_perfect_score": outputs[2]["score"], "all_negative_false_positive_score": outputs[3]["score"]}


def parse_historical_rows(markdown: str) -> dict[tuple[str, str, int], dict[str, float]]:
    headings = list(re.finditer(r"^#### (test_none|test_soft|test_hard)\s*$", markdown, re.MULTILINE))
    parsed: dict[tuple[str, str, int], dict[str, float]] = {}
    for index, heading in enumerate(headings):
        protocol = heading.group(1)
        section = markdown[heading.end() : headings[index + 1].start() if index + 1 < len(headings) else len(markdown)]
        if "| Method | Seed |" not in section:
            continue
        rows = re.findall(r"^\| (equal-parameter shared fusion) \| (42|43|44|45|46) \| ([0-9.]+)/([0-9.]+)/([0-9.]+) \| ([0-9.]+)/([0-9.]+)/([0-9.]+) \| ([0-9.]+) \|", section, re.MULTILINE)
        if len(rows) != 5:
            fail(f"historical {protocol}: expected five shared rows, found {len(rows)}")
        for method, seed_text, duar, dmf1, dscore, puar, pmf1, pscore, historical_mean in rows:
            key = (protocol, method, int(seed_text))
            if key in parsed:
                fail(f"duplicate historical row {key}")
            parsed[key] = {"d_uar": float(duar), "d_mf1": float(dmf1), "d_score": float(dscore), "p_uar": float(puar), "p_mf1": float(pmf1), "p_score": float(pscore), "mean": float(historical_mean)}
    expected = {(p, "equal-parameter shared fusion", s) for p in PROTOCOLS for s in SEEDS}
    if set(parsed) != expected:
        fail(f"historical rows are incomplete: missing={sorted(expected - set(parsed))}, extra={sorted(set(parsed) - expected)}")
    return parsed


def summarize(values: list[float]) -> dict[str, float]:
    avg, sd = mean(values), stdev(values)
    return {"mean": avg, "sample_sd": sd, "min": min(values), "max": max(values), "range": max(values) - min(values), "ci95_low": avg - T_MULTIPLIER * sd / math.sqrt(len(values)), "ci95_high": avg + T_MULTIPLIER * sd / math.sqrt(len(values))}


def aggregate_series(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = {
        task: {metric: [row["metrics"][task][metric] for row in rows] for metric in ("uar", "mf1", "score")}
        for task in TASKS
    }
    values["mean"] = {"mean": [row["metrics"]["mean"] for row in rows]}
    return {
        task: {metric: summarize(series) for metric, series in metric_values.items()}
        for task, metric_values in values.items()
    }


def aggregates(dev_rows: list[dict[str, Any]], rows: dict[tuple[int, str], dict[str, Any]], historical: dict[tuple[str, str, int], dict[str, float]]) -> dict[str, Any]:
    output: dict[str, Any] = {"dev": aggregate_series(dev_rows)}
    for protocol in PROTOCOLS:
        output[protocol] = aggregate_series([rows[(seed, protocol)] for seed in SEEDS])
        direct = [rows[(seed, protocol)]["metrics"]["mean"] - historical[(protocol, "equal-parameter shared fusion", seed)]["mean"] for seed in SEEDS]
        output[protocol]["paired_mean_delta"] = {**summarize(direct), "deltas": direct, "wins": sum(value > 0 for value in direct), "wins_over_5": f"{sum(value > 0 for value in direct)}/5"}
        decomposition = {}
        for task, key in (("depression", "d_score"), ("parkinson", "p_score")):
            deltas = [rows[(seed, protocol)]["metrics"][task]["score"] - historical[(protocol, "equal-parameter shared fusion", seed)][key] for seed in SEEDS]
            decomposition[task] = {"score_deltas": deltas, "mean_contribution_deltas": [value / 2 for value in deltas]}
        residual = [direct[i] - (decomposition["depression"]["mean_contribution_deltas"][i] + decomposition["parkinson"]["mean_contribution_deltas"][i]) for i in range(5)]
        output[protocol]["task_decomposition"] = {**decomposition, "direct_minus_sum_rounding_residual": residual}
    return output


def audit(dev_path: Path, test_path: Path, marker_path: Path, stage7_path: Path, repo: Path) -> dict[str, Any]:
    validate_artifact_identity(dev_path, "dev")
    validate_artifact_identity(test_path, "test")
    validate_artifact_identity(marker_path, "marker")
    marker = marker_path.read_text()
    if "COMPLETED" not in marker or '"evaluation_count": 15' not in marker:
        fail("marker does not retain COMPLETED/evaluation_count=15")
    dev, test = json.loads(dev_path.read_text()), json.loads(test_path.read_text())
    validate_dev(dev, repo)
    rows = validate_test(test, dev)
    historical = parse_historical_rows(stage7_path.read_text())
    aggregates_result = aggregates(dev["results"], rows, historical)
    return {"status": "OFFLINE_CHECKS_PASS", "verdict": "PENDING MANAGER REVIEW", "artifact_hashes": {"dev": sha256(dev_path), "test": sha256(test_path), "marker": sha256(marker_path)}, "artifact_bytes": {"dev": dev_path.stat().st_size, "test": test_path.stat().st_size, "marker": marker_path.stat().st_size}, "independent_oracle": explicit_metric_oracle(), "historical_rows": len(historical), "aggregates": aggregates_result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--stage7", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dev, args.test, args.marker, args.stage7, args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
