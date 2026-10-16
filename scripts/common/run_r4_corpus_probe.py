"""TASK-006B: frozen R4 corpus-identity probe on canonical TRAIN -> DEV."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from chimera_ml.core.batch import Batch
from fusion.data.wsm_ramps_semantic_datamodule import WSMRampsSemanticDataModule
from chimera_ml.core.registry import MODELS
import fusion.models.av_r3_disease_query  # noqa: F401


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("/media/maxim/Programs/Features/WSM/stage6_corpus_probe/r4_corpus_probe_v1.json")
CHECKPOINTS = [
    {
        "seed": 42,
        "config": ROOT / "configs/wsm_mm_pd_dep_v1/fusion/37_r4_ra_stch_optuna_selected_seed42.yaml",
        "checkpoint": ROOT / "logs/wsm_mm_pd_dep_v1/optuna_r4_ra_stch_base_seed42_2026-09-28_01-54_wsm_av_r3_disease_query_model_r4-ra-stch-optuna-v1-18c6-012_0ce9c7a4/checkpoints/epoch=11_dev_mean_score=0.8191.pt",
        "sha256": "104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a",
        "dev_mean": 0.8191424538,
    },
    {
        "seed": 43,
        "config": ROOT / "configs/wsm_mm_pd_dep_v1/fusion/38_r4_ra_stch_optuna_confirm_seed43.yaml",
        "checkpoint": ROOT / "logs/wsm_mm_pd_dep_v1/r4_ra_stch_optuna_trial012_confirm_seed43_2026-09-28_12-01_wsm_av_r3_disease_query_model_12038580/checkpoints/epoch=10_dev_mean_score=0.7614.pt",
        "sha256": "6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e",
        "dev_mean": 0.761445,
    },
    {
        "seed": 44,
        "config": ROOT / "configs/wsm_mm_pd_dep_v1/fusion/39_r4_ra_stch_optuna_confirm_seed44.yaml",
        "checkpoint": ROOT / "logs/wsm_mm_pd_dep_v1/r4_ra_stch_optuna_trial012_confirm_seed44_2026-09-28_12-07_wsm_av_r3_disease_query_model_41bc602b/checkpoints/epoch=11_dev_mean_score=0.7801.pt",
        "sha256": "b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17",
        "dev_mean": 0.780139,
    },
]
REP_DIMS = {"audio_projected": 160, "video_projected": 160, "projected_av": 320, "task_fused": 320, "gates": 4, "logits": 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoints() -> list[dict[str, Any]]:
    for item in CHECKPOINTS:
        if not item["config"].is_file() or not item["checkpoint"].is_file():
            raise RuntimeError(f"missing frozen config/checkpoint for seed {item['seed']}")
        actual = sha256(item["checkpoint"])
        if actual != item["sha256"]:
            raise RuntimeError(f"checkpoint SHA256 mismatch for seed {item['seed']}: {actual}")
    return CHECKPOINTS


def _config_parts(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["model"]["name"] != "wsm_av_r3_disease_query_model":
        raise RuntimeError("unexpected frozen model registry key")
    if config["data"]["name"] != "wsm_ramps_semantic_datamodule":
        raise RuntimeError("unexpected frozen datamodule registry key")
    return dict(config["model"]["params"]), dict(config["data"]["params"])


def _loader(dataset: Any, collate_fn: Any) -> DataLoader:
    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_fn)


def _corpus_labels(meta: list[dict[str, Any]]) -> torch.Tensor:
    values = []
    for sample_meta in meta:
        corpus = sample_meta.get("corpus")
        if corpus not in {"depression", "parkinson"}:
            raise RuntimeError(f"invalid canonical sample_meta corpus: {corpus!r}")
        values.append(0 if corpus == "depression" else 1)
    return torch.tensor(values, dtype=torch.float64)


def extract(model: torch.nn.Module, dm: WSMRampsSemanticDataModule, limit: int | None = None) -> dict[str, dict[str, Any]]:
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in REP_DIMS}
    ids: list[str] = []
    labels: list[torch.Tensor] = []
    seen_test_iteration = False
    for dataset in (dm.train_dataset, dm.val_dataset):
        for batch in _loader(dataset, dm.collate_fn):
            with torch.no_grad():
                output = model(batch)
            task_features = output.aux["task_features"]
            gates = output.aux["task_modality_weights"]
            values = {
                "audio_projected": output.aux["features_audio"],
                "video_projected": output.aux["features_video"],
                "projected_av": torch.cat((output.aux["features_audio"], output.aux["features_video"]), dim=1),
                "task_fused": task_features.reshape(task_features.shape[0], -1),
                "gates": gates.reshape(gates.shape[0], -1),
                "logits": output.preds,
            }
            sample_meta = batch.meta["sample_meta"]
            ids.extend(str(meta["segment_id"]) for meta in sample_meta)
            labels.append(_corpus_labels(sample_meta))
            for name, value in values.items():
                if tuple(value.shape[1:]) != (REP_DIMS[name],) or not torch.isfinite(value).all():
                    raise RuntimeError(f"representation invariant failed for {name}: {tuple(value.shape)}")
                collected[name].append(value.detach().cpu().to(torch.float64))
            if limit is not None and len(ids) >= limit:
                break
        if limit is not None and len(ids) >= limit:
            break
    if seen_test_iteration:
        raise RuntimeError("Test loader was iterated")
    split = "train" if len(ids) == len(dm.train_dataset) else "dev"
    # The caller invokes this once per split; this guard catches accidental mixed datasets.
    result = {name: torch.cat(values, dim=0) for name, values in collected.items()}
    result["segment_ids"] = ids  # type: ignore[assignment]
    result["corpus_labels"] = torch.cat(labels, dim=0)  # type: ignore[assignment]
    result["split"] = split  # type: ignore[assignment]
    return result


def extract_split(model: torch.nn.Module, dataset: Any, collate_fn: Any, expected: int) -> dict[str, Any]:
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in REP_DIMS}
    ids: list[str] = []
    labels: list[torch.Tensor] = []
    for batch in _loader(dataset, collate_fn):
        with torch.no_grad():
            output = model(batch)
        task_features = output.aux["task_features"]
        gates = output.aux["task_modality_weights"]
        values = {
            "audio_projected": output.aux["features_audio"],
            "video_projected": output.aux["features_video"],
            "projected_av": torch.cat((output.aux["features_audio"], output.aux["features_video"]), dim=1),
            "task_fused": task_features.reshape(task_features.shape[0], -1),
            "gates": gates.reshape(gates.shape[0], -1),
            "logits": output.preds,
        }
        sample_meta = batch.meta["sample_meta"]
        ids.extend(str(meta["segment_id"]) for meta in sample_meta)
        labels.append(_corpus_labels(sample_meta))
        for name, value in values.items():
            if tuple(value.shape[1:]) != (REP_DIMS[name],) or not torch.isfinite(value).all():
                raise RuntimeError(f"representation invariant failed for {name}: {tuple(value.shape)}")
            collected[name].append(value.detach().cpu().to(torch.float64))
    if len(ids) != expected or len(set(ids)) != expected:
        raise RuntimeError(f"unexpected extraction count/uniqueness: {len(ids)} / {expected}")
    order = sorted(range(len(ids)), key=ids.__getitem__)
    ordered_ids = [ids[i] for i in order]
    ordered_labels = torch.cat(labels)[order]
    return {name: torch.cat(values)[order] for name, values in collected.items()} | {"segment_ids": ordered_ids, "corpus_labels": ordered_labels}


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores, labels = scores.detach().to(torch.float64), labels.detach().to(torch.float64)
    positives, negatives = labels == 1, labels == 0
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    if not n_pos or not n_neg:
        raise ValueError("AUROC requires both classes")
    order = torch.argsort(scores, descending=False, stable=True)
    sorted_scores, sorted_labels = scores[order], labels[order]
    ranks = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    # Midranks for ties, followed by the Mann-Whitney U statistic.
    rank_sum = torch.zeros_like(ranks)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        rank_sum[start:end] = (ranks[start] + ranks[end - 1]) / 2
        start = end
    positive_rank_sum = rank_sum[sorted_labels == 1].sum()
    return float((positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    predicted = probs >= 0.5
    positives, negatives = labels == 1, labels == 0
    tpr = (predicted[positives] == 1).double().mean()
    tnr = (predicted[negatives] == 0).double().mean()
    return {"auroc": auroc(probs, labels), "balanced_accuracy": float((tpr + tnr) / 2), "brier": float(((probs - labels) ** 2).mean())}


def fit_probe(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    mean, std = x.mean(0), x.std(0, correction=0)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    train_x = (x - mean) / std
    n = len(y)
    counts = torch.bincount(y.to(torch.long), minlength=2).to(torch.float64)
    sample_weights = torch.where(y == 0, n / (2 * counts[0]), n / (2 * counts[1]))
    weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], lr=1.0, max_iter=250, tolerance_grad=1e-10, tolerance_change=1e-12, history_size=50, line_search_fn="strong_wolfe")
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = train_x @ weight + bias
        objective = (torch.nn.functional.binary_cross_entropy_with_logits(logits, y, weight=sample_weights, reduction="sum") / sample_weights.sum()) + 1e-4 * (weight @ weight)
        objective.backward()
        return objective
    optimizer.step(closure)
    with torch.no_grad():
        logits = train_x @ weight + bias
        objective = float((torch.nn.functional.binary_cross_entropy_with_logits(logits, y, weight=sample_weights, reduction="sum") / sample_weights.sum()) + 1e-4 * (weight @ weight))
    return torch.cat((weight.detach(), bias.detach().reshape(1))), {"train_objective": objective, "train_mean": mean.tolist(), "train_std": std.tolist()}


def predict(probe: torch.Tensor, train_stats: dict[str, Any], x: torch.Tensor) -> torch.Tensor:
    mean, std = torch.tensor(train_stats["train_mean"], dtype=torch.float64), torch.tensor(train_stats["train_std"], dtype=torch.float64)
    return ((x - mean) / std) @ probe[:-1] + probe[-1]


def synthetic_checks() -> None:
    assert auroc(torch.tensor([0., 1., 2., 3.]), torch.tensor([0., 0., 1., 1.])) == 1.0
    assert auroc(torch.tensor([3., 2., 1., 0.]), torch.tensor([0., 0., 1., 1.])) == 0.0
    assert auroc(torch.ones(4), torch.tensor([0., 0., 1., 1.])) == 0.5
    x = torch.tensor([[0.], [1.], [2.], [3.]], dtype=torch.float64)
    y = torch.tensor([0., 0., 1., 1.], dtype=torch.float64)
    probe, stats = fit_probe(x, y)
    assert torch.isfinite(probe).all() and math.isfinite(stats["train_objective"])


def gate_audit(dev: dict[str, Any]) -> dict[str, Any]:
    gates = dev["gates"].reshape(-1, 2, 2)
    labels = dev["corpus_labels"]
    result = {}
    for task, task_name in enumerate(("depression_query", "parkinson_query")):
        result[task_name] = {}
        for corpus, value in (("depression", 0), ("parkinson", 1)):
            audio = gates[labels == value, task, 0]
            result[task_name][corpus] = {"mean": float(audio.mean()), "sample_std": float(audio.std(unbiased=True)), "q25": float(torch.quantile(audio, .25)), "median": float(torch.quantile(audio, .5)), "q75": float(torch.quantile(audio, .75))}
        result[task_name]["mean_audio_weight_depression_minus_parkinson"] = result[task_name]["depression"]["mean"] - result[task_name]["parkinson"]["mean"]
    return result


def run(firewall: bool) -> dict[str, Any] | None:
    verify_checkpoints()
    synthetic_checks()
    audit: dict[str, Any] = {"checkpoint_loads": [], "splits": {}, "representations": REP_DIMS, "no_test_loader_iteration": True}
    extracted: dict[int, dict[str, dict[str, Any]]] = {}
    for item in CHECKPOINTS:
        model_params, data_params = _config_parts(item["config"])
        data_params.update(batch_size=32, num_workers=0, pin_memory=False, persistent_workers=False, shuffle_train=False, drop_last_train=False)
        dm = WSMRampsSemanticDataModule(**data_params)
        model = MODELS.get("wsm_av_r3_disease_query_model")(**model_params)
        payload = torch.load(item["checkpoint"], map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        audit["checkpoint_loads"].append({"seed": item["seed"], "epoch": payload["epoch"], "sha256": item["sha256"], "model_eval": not model.training})
        train_dataset = Subset(dm.train_dataset, range(2)) if firewall else dm.train_dataset
        dev_dataset = Subset(dm.val_dataset, range(2)) if firewall else dm.val_dataset
        train = extract_split(model, train_dataset, dm.collate_fn, 2 if firewall else 6325)
        dev = extract_split(model, dev_dataset, dm.collate_fn, 2 if firewall else 933)
        extracted[item["seed"]] = {"train": train, "dev": dev}
        audit["splits"][str(item["seed"])] = {"train": {"count": 2 if firewall else 6325, "corpus_counts": {"depression": int((train["corpus_labels"] == 0).sum()), "parkinson": int((train["corpus_labels"] == 1).sum())}}, "dev": {"count": 2 if firewall else 933, "corpus_counts": {"depression": int((dev["corpus_labels"] == 0).sum()), "parkinson": int((dev["corpus_labels"] == 1).sum())}}}
        if firewall:
            continue
    if firewall:
        return audit
    metrics: dict[str, Any] = {}
    for item in CHECKPOINTS:
        seed, train, dev = item["seed"], extracted[item["seed"]]["train"], extracted[item["seed"]]["dev"]
        seed_result: dict[str, Any] = {}
        generator = torch.Generator().manual_seed(9201 + seed)
        shuffled_y = train["corpus_labels"][torch.randperm(len(train["corpus_labels"]), generator=generator)]
        if torch.equal(shuffled_y, train["corpus_labels"]):
            raise RuntimeError("shuffled labels equal original labels")
        for name in REP_DIMS:
            probe, stats = fit_probe(train[name], train["corpus_labels"])
            shuffled_probe, shuffled_stats = fit_probe(train[name], shuffled_y)
            true_metrics = _metrics(predict(probe, stats, dev[name]), dev["corpus_labels"])
            shuffled_metrics = _metrics(predict(shuffled_probe, shuffled_stats, dev[name]), dev["corpus_labels"])
            seed_result[name] = {"true": true_metrics | stats, "shuffled": shuffled_metrics | shuffled_stats, "true_minus_shuffled_auroc": true_metrics["auroc"] - shuffled_metrics["auroc"]}
        seed_result["gate_audit"] = gate_audit(dev)
        metrics[str(seed)] = seed_result
    summaries = {}
    for name in REP_DIMS:
        for control in ("true", "shuffled"):
            values = [metrics[str(item["seed"])][name][control] for item in CHECKPOINTS]
            summaries[f"{name}/{control}"] = {metric: {"mean": float(torch.tensor([v[metric] for v in values]).mean()), "sample_std": float(torch.tensor([v[metric] for v in values]).std(unbiased=True))} for metric in ("auroc", "balanced_accuracy", "brier")}
    fused_deltas = [metrics[str(item["seed"])] ["task_fused"]["true"]["auroc"] - metrics[str(item["seed"])] ["projected_av"]["true"]["auroc"] for item in CHECKPOINTS]
    fused_true = [metrics[str(item["seed"])] ["task_fused"]["true"]["auroc"] for item in CHECKPOINTS]
    shuffled = [metrics[str(item["seed"])] ["task_fused"]["shuffled"]["auroc"] for item in CHECKPOINTS]
    strong = sum(v >= .9 for v in fused_true) >= 2 and sum(a - b >= .2 for a, b in zip(fused_true, shuffled)) >= 2
    amplified = sum(v >= .05 for v in fused_deltas) >= 2 and sum(fused_deltas) / 3 >= .05
    audit.update({"probe": metrics, "three_seed_summaries": summaries, "task_fused_true_minus_shuffled_auroc": [a - b for a, b in zip(fused_true, shuffled)], "task_fused_minus_projected_av_auroc": fused_deltas, "gate_audit": {str(item["seed"]): metrics[str(item["seed"])] ["gate_audit"] for item in CHECKPOINTS}, "interpretation": {"strong_corpus_decodability": "CORPUS IDENTITY IS STRONGLY LINEARLY DECODEABLE FROM R4 FUSED REPRESENTATIONS" if strong else "STRONG CORPUS-ID DECODABILITY FLAG NOT TRIGGERED", "fusion_amplification": "R4 TASK-CONDITIONED FUSION AMPLIFIES LINEAR CORPUS DECODABILITY" if amplified else "FUSION AMPLIFICATION OF CORPUS DECODABILITY NOT SUPPORTED"}, "structural_boundary": "Corpus identity is structurally coupled to which disease label is observed; this is a risk diagnostic, not causal proof of shortcut use.", "no_causal_shortcut_claim": True, "no_test_loader_iteration": True, "no_main_model_training": True})
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--firewall", action="store_true")
    args = parser.parse_args()
    result = run(args.firewall)
    if args.firewall:
        print(json.dumps(result, sort_keys=True))
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
