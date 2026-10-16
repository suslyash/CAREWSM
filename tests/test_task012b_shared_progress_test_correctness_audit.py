import json
import math
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from common.callbacks.wsm_segment_callback import compute_sparse_two_task_metrics
from scripts.common.audit_task012a_saved_evidence import (
    PROTOCOLS,
    SEEDS,
    aggregates,
    oracle_metrics,
    parse_historical_rows,
    summarize,
    validate_artifact_identity,
    validate_dev,
    validate_metric_block,
    validate_test,
)


def production_task(metric, task):
    prefix = f"synthetic/{task}"
    return {
        key: metric[f"{prefix}/{key}"]
        for key in ("num_samples", "tn", "fp", "fn", "tp", "uar", "mf1", "score")
    }


def test_oracle_matches_unchanged_production_metric_on_sparse_nan_fixture():
    logits = torch.tensor([[0.0, -1e-12], [1e-12, 2.0], [-2.0, 0.0], [3.0, -3.0], [4.0, 4.0]])
    targets = torch.tensor([[0.0, float("nan")], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [float("nan"), 1.0]])
    observed = torch.tensor([[True, False], [True, True], [True, False], [True, True], [False, True]])
    production = compute_sparse_two_task_metrics(logits, targets, observed, "synthetic")
    assert production["synthetic/depression/num_samples"] == 4
    assert production["synthetic/parkinson/num_samples"] == 3
    for task_id, task in enumerate(("depression", "parkinson")):
        selected = observed[:, task_id]
        expected = oracle_metrics(logits[selected, task_id].tolist(), targets[selected, task_id].tolist())
        actual = production_task(production, task)
        assert actual == expected
    assert production["synthetic/depression/tn"] == 0
    assert production["synthetic/depression/fp"] == 2
    assert production["synthetic/depression/fn"] == 1
    assert production["synthetic/depression/tp"] == 1
    assert production["synthetic/parkinson/tn"] == 1
    assert production["synthetic/parkinson/fp"] == 0
    assert production["synthetic/parkinson/fn"] == 0
    assert production["synthetic/parkinson/tp"] == 2
    assert production["synthetic/depression/uar"] == pytest.approx(0.25)
    assert production["synthetic/depression/mf1"] == pytest.approx(0.20)
    assert production["synthetic/depression/score"] == pytest.approx(0.225)
    assert production["synthetic/mean_score"] == pytest.approx(0.6125)
    assert production["synthetic/mean_score"] == pytest.approx(
        (production["synthetic/depression/score"] + production["synthetic/parkinson/score"]) / 2
    )


def test_metric_edge_contract_and_zero_boundary():
    perfect_negative = compute_sparse_two_task_metrics(
        torch.tensor([[-1.0, -1.0], [-1.0, -1.0]]),
        torch.tensor([[0.0, 0.0], [0.0, float("nan")]]),
        torch.tensor([[True, True], [True, False]]),
        "edge",
    )
    assert perfect_negative["edge/depression/uar"] == 1.0
    assert oracle_metrics([-1.0, -1.0], [0, 0])["score"] == 1.0
    with pytest.raises(ValueError):
        compute_sparse_two_task_metrics(torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.bool), "empty")
    with pytest.raises(ValueError):
        compute_sparse_two_task_metrics(torch.zeros(2, 2), torch.tensor([[0.0, 2.0], [0.0, 2.0]]), torch.ones(2, 2, dtype=torch.bool), "bad-target")
    with pytest.raises(ValueError):
        compute_sparse_two_task_metrics(torch.zeros(2, 2), torch.zeros(2, 2), torch.ones(2, 2), "bad-mask")
    with pytest.raises(ValueError):
        oracle_metrics([], [])


def test_global_concat_differs_from_unweighted_batch_average():
    batch_a = (torch.tensor([[-2.0, -2.0], [2.0, 2.0]]), torch.tensor([[0.0, 0.0], [1.0, 1.0]]), torch.ones(2, 2, dtype=torch.bool))
    batch_b = (torch.tensor([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]), torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), torch.ones(3, 2, dtype=torch.bool))
    first = compute_sparse_two_task_metrics(*batch_a, "batch")
    second = compute_sparse_two_task_metrics(*batch_b, "batch")
    concatenated = compute_sparse_two_task_metrics(
        torch.cat([batch_a[0], batch_b[0]]), torch.cat([batch_a[1], batch_b[1]]), torch.cat([batch_a[2], batch_b[2]]), "batch"
    )
    average_of_batches = (first["batch/mean_score"] + second["batch/mean_score"]) / 2
    assert len(batch_a[0]) == 2 and len(batch_b[0]) == 3
    assert concatenated["batch/depression/tn"] == 1
    assert concatenated["batch/depression/fp"] == 3
    assert concatenated["batch/depression/fn"] == 0
    assert concatenated["batch/depression/tp"] == 1
    assert concatenated["batch/depression/uar"] == pytest.approx(0.625)
    assert concatenated["batch/depression/mf1"] == pytest.approx(0.4)
    assert concatenated["batch/mean_score"] == pytest.approx(0.5125)
    assert average_of_batches == pytest.approx(0.5)
    assert concatenated["batch/mean_score"] != average_of_batches


def test_nonconstant_student_ci_rejects_ddof_zero_and_wrong_multiplier():
    values = [0.7, 0.8, 0.9, 1.0, 1.1]
    result = summarize(values)
    expected_sd = math.sqrt(0.025)
    expected_half_width = 2.7764451051977987 * expected_sd / math.sqrt(5)
    assert result["sample_sd"] == pytest.approx(expected_sd)
    assert result["ci95_low"] == pytest.approx(0.9 - expected_half_width)
    assert result["ci95_high"] == pytest.approx(0.9 + expected_half_width)
    assert result["sample_sd"] != pytest.approx(math.sqrt(0.02))


def test_historical_parser_joins_by_explicit_protocol_heading_and_rejects_duplicates():
    source = Path("docs/STAGE7_FINAL_TEST_EVIDENCE_EN.md").read_text()
    headings = list(__import__("re").finditer(r"^#### (test_none|test_soft|test_hard)\s*$", source, __import__("re").MULTILINE))
    blocks = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(source)
        block = source[heading.start():end]
        if "| Method | Seed |" in block:
            blocks.append(block)
    reordered = "\n".join(reversed(blocks))
    parsed = parse_historical_rows(reordered)
    assert len(parsed) == 15
    assert {protocol for protocol, _, _ in parsed} == set(PROTOCOLS)
    with pytest.raises(ValueError):
        parse_historical_rows(reordered + "\n" + blocks[0])
    with pytest.raises(ValueError):
        parse_historical_rows(reordered.replace("| equal-parameter shared fusion | 46 |", "| missing method | 46 |", 1))


def test_structural_negative_fixtures_reject_duplicate_rows_and_empty_dev():
    dev_path = Path("logs/task012a_postclosure_shared_progress_test/dev_preflight.json")
    test_path = Path("logs/task012a_postclosure_shared_progress_test/test_results.json")
    dev = json.loads(dev_path.read_text())
    test = json.loads(test_path.read_text())
    repo = Path.cwd()
    duplicate = json.loads(json.dumps(test))
    duplicate["results"].append(json.loads(json.dumps(duplicate["results"][0])))
    with pytest.raises(ValueError, match="exactly 15"):
        validate_test(duplicate, dev)
    empty_dev = json.loads(json.dumps(dev))
    empty_dev["results"] = []
    empty_dev["ledger"] = []
    with pytest.raises(ValueError, match="exactly five"):
        validate_dev(empty_dev, repo)


def test_validator_rejects_hash_nan_range_count_and_identity_failures():
    test = json.loads(Path("logs/task012a_postclosure_shared_progress_test/test_results.json").read_text())
    dev = json.loads(Path("logs/task012a_postclosure_shared_progress_test/dev_preflight.json").read_text())
    with tempfile.TemporaryDirectory() as directory:
        bad = Path(directory) / "test_results.json"
        shutil.copyfile("logs/task012a_postclosure_shared_progress_test/test_results.json", bad)
        bad.write_bytes(bad.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="identity mismatch"):
            validate_artifact_identity(bad, "test")
    invalid_metrics = json.loads(json.dumps(test["results"][0]["metrics"]))
    invalid_metrics["depression"]["uar"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_metric_block(invalid_metrics, "bad/nan", 1364, {"depression": 827, "parkinson": 537})
    invalid_metrics = json.loads(json.dumps(test["results"][0]["metrics"]))
    invalid_metrics["observed_counts"]["depression"] = 828
    with pytest.raises(ValueError, match="observed count"):
        validate_metric_block(invalid_metrics, "bad/count", 1364, {"depression": 827, "parkinson": 537})
    bad_identity = json.loads(json.dumps(dev))
    bad_identity["ledger"][0]["run_name"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        validate_dev(bad_identity, Path.cwd())


def test_metric_aggregate_contract_has_all_seven_series():
    dev = json.loads(Path("logs/task012a_postclosure_shared_progress_test/dev_preflight.json").read_text())
    test = json.loads(Path("logs/task012a_postclosure_shared_progress_test/test_results.json").read_text())
    rows = {(row["seed"], row["protocol"]): row for row in test["results"]}
    historical = parse_historical_rows(Path("docs/STAGE7_FINAL_TEST_EVIDENCE_EN.md").read_text())
    result = aggregates(dev["results"], rows, historical)
    for protocol in ("dev", *PROTOCOLS):
        assert set(result[protocol]["depression"]) == {"uar", "mf1", "score"}
        assert set(result[protocol]["parkinson"]) == {"uar", "mf1", "score"}
        assert set(result[protocol]["mean"]) == {"mean"}
        for task in ("depression", "parkinson"):
            for metric in ("uar", "mf1", "score"):
                assert set(result[protocol][task][metric]) == {"mean", "sample_sd", "min", "max", "range", "ci95_low", "ci95_high"}
        assert set(result[protocol]["mean"]["mean"]) == {"mean", "sample_sd", "min", "max", "range", "ci95_low", "ci95_high"}
    assert result["dev"]["mean"]["mean"]["mean"] == pytest.approx(0.801825630372)
    assert result["test_none"]["mean"]["mean"]["ci95_low"] == pytest.approx(0.730996720645)
    assert result["test_none"]["depression"]["uar"]["mean"] == pytest.approx(0.7664, abs=0.0001)
    for protocol in PROTOCOLS:
        assert result[protocol]["paired_mean_delta"]["wins_over_5"] == "0/5"
