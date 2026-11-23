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
