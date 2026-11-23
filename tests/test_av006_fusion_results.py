import importlib.util
from pathlib import Path


MODULE_PATH = Path("scripts/fusion/collect_results.py")
SPEC = importlib.util.spec_from_file_location("collect_results", MODULE_PATH)
collector = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(collector)


def test_expected_matrix_has_285_unique_configurations():
    configs = sorted(Path("configs/fusion/sweep").glob("*.yaml"))
    identities = [collector.parse_config(path)["run_name"] for path in configs]
    assert len(identities) == 285
    assert len(set(identities)) == 285


def test_runtime_ambiguity_is_not_silently_resolved(tmp_path):
    runtime_root = tmp_path / "logs/fusion/sweep/sweep"
    (runtime_root / "run_20260101").mkdir(parents=True)
    (runtime_root / "run_20260102").mkdir(parents=True)
    assert len(collector.runtime_candidates(tmp_path, "run")) == 2


def test_checkpoint_requires_one_non_last_file_and_matching_epoch(tmp_path):
    directory = tmp_path / "run"
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "last.pt").write_bytes(b"last")
    (checkpoints / "epoch=3_val_mean_corpora_tasks_macro_recall=0.8123.pt").write_bytes(b"x")
    path, epoch, value, error = collector.checkpoint(directory)
    assert path is not None and epoch == 3 and value == 0.8123 and error is None


def test_overall_uar_requires_all_four_components():
    values = {f"val_{component}_macro_recall": 0.5 for component in collector.COMPONENTS}
    assert collector.overall(values, "val") == 0.5
    values.pop("val_wsm_parkinson_macro_recall")
    assert collector.overall(values, "val") is None


def test_sample_standard_deviation_uses_ddof_one():
    assert collector.statistics.stdev([1.0, 2.0, 3.0]) == 1.0


def test_incomplete_variant_is_not_eligible_for_ranking():
    rows = [{"experiment_id": "x", "status": "COMPLETED", "val_overall_uar": 0.9, "test_overall_uar": 0.8}]
    rows[0].update({field: 0.5 for field in collector.metric_fields()})
    summary = collector.aggregate(rows)[0]
    assert summary["completed_folds"] == 1
    assert summary["five_fold_complete"] == "no"
    assert summary["val_overall_uar_mean"] == ""


def test_atomic_write_replaces_existing_file(tmp_path):
    path = tmp_path / "reports" / "result.csv"
    collector.atomic_write(path, "first\n")
    collector.atomic_write(path, "second\n")
    assert path.read_text() == "second\n"


def test_audio_vs_text_guided_keeps_val_pairs_when_test_is_missing():
    common = {
        "fold": "0",
        "status": "MISSING_METRICS",
        "val_overall_uar": 0.7,
        "test_overall_uar": "",
    }
    records = [
        {**common, "experiment_id": "text_guided__A1_V4_T9", "architecture": "text_guided"},
        {**common, "experiment_id": "audio_guided__A1_V4_T9", "architecture": "audio_guided",
         "val_overall_uar": 0.8},
    ]
    result = collector.compare_guided(records)
    assert len(result) == 5
    observed = next(row for row in result if row["fold"] == "0")
    assert observed["text_guided_status"] is True
    assert observed["audio_guided_status"] is True
    assert abs(observed["val_overall_uar_audio_minus_text"] - 0.1) < 1e-12
    assert observed["test_overall_uar_audio_minus_text"] == ""


def test_recovery_requires_checkpoint_and_config_identity(tmp_path):
    config = collector.parse_config(
        Path("configs/fusion/sweep/audio_guided__A1_V4_T10__fold0.yaml")
    )
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"selected")
    record = {
        "run_name": config["run_name"],
        "model": config["model"],
        "fold": config["fold"],
        "config_sha256": config["config_sha256"],
        "checkpoint_sha256": collector.sha256(checkpoint),
        "selected_epoch": 4,
        "selected_val_monitor": 0.6747,
        "provenance": "post-training inference from the original selected checkpoint",
    }
    record.update(
        {
            f"test_{component}_{metric}": 0.5
            for component in collector.COMPONENTS
            for metric in collector.METRICS
        }
    )
    assert collector.valid_recovery(record, config, checkpoint, 4, 0.6747)
    record["checkpoint_sha256"] = "wrong"
    assert not collector.valid_recovery(record, config, checkpoint, 4, 0.6747)


def test_duplicate_recovery_records_are_rejected(tmp_path):
    path = tmp_path / "test_results.jsonl"
    path.write_text(
        '{"run_name":"run","provenance":"one"}\n'
        '{"run_name":"run","provenance":"two"}\n',
        encoding="utf-8",
    )
    records, ambiguous = collector.load_recovery(path)
    assert "run" in records
    assert ambiguous == {"run"}
