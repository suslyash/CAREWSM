from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from chimera_ml.callbacks._utils import resolve_splits
from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import CALLBACKS
from chimera_ml.core.types import ModelOutput
from chimera_ml.metrics.confusion_matrix_metric import ConfusionMatrixMetric
from chimera_ml.training.cached_split_outputs import CachedSplitOutputs


def _class_recall(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _class_f1(true_positive: int, false_positive: int, false_negative: int) -> float | None:
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else None


def compute_sparse_two_task_metrics(logits: torch.Tensor, targets: torch.Tensor, observed_mask: torch.Tensor, prefix: str = 'dev', task_names: tuple[str, str] = ('depression', 'parkinson')) -> dict[str, float]:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or tuple(logits.shape[1:]) != (2,):
        raise ValueError('logits must have shape [N,2]')
    if not isinstance(targets, torch.Tensor) or tuple(targets.shape) != tuple(logits.shape):
        raise ValueError('targets must have shape [N,2]')
    if not isinstance(observed_mask, torch.Tensor) or observed_mask.dtype != torch.bool or tuple(observed_mask.shape) != tuple(logits.shape):
        raise ValueError('observed_mask must be bool [N,2]')
    if len(task_names) != 2:
        raise ValueError('task_names must contain two names')
    output: dict[str, float] = {}
    scores: list[float] = []
    logits = logits.detach().cpu()
    targets = targets.detach().cpu()
    observed_mask = observed_mask.detach().cpu()
    for task_id, task_name in enumerate(task_names):
        selected = observed_mask[:, task_id]
        count = int(selected.sum().item())
        if count == 0:
            raise ValueError(f'{prefix}/{task_name} has zero observed samples')
        task_targets = targets[selected, task_id]
        if not bool(torch.isfinite(task_targets).all()):
            raise ValueError(f'{prefix}/{task_name} observed targets must be finite')
        if not bool(torch.logical_or(task_targets == 0, task_targets == 1).all()):
            raise ValueError(f'{prefix}/{task_name} observed targets must be binary 0/1')
        predictions = logits[selected, task_id] >= 0.0
        positives = task_targets == 1
        negatives = task_targets == 0
        tp = int((predictions & positives).sum().item())
        fp = int((predictions & negatives).sum().item())
        fn = int((~predictions & positives).sum().item())
        tn = int((~predictions & negatives).sum().item())
        recall_negative = _class_recall(tn, tn + fp)
        recall_positive = _class_recall(tp, tp + fn)
        f1_negative = _class_f1(tn, fn, fp)
        f1_positive = _class_f1(tp, fp, fn)
        recalls = [value for value in (recall_negative, recall_positive) if value is not None]
        f1_values = [value for value in (f1_negative, f1_positive) if value is not None]
        uar = float(sum(recalls) / len(recalls))
        mf1 = float(sum(f1_values) / len(f1_values))
        score = (uar + mf1) / 2.0
        root = f'{prefix}/{task_name}'
        output.update({f'{root}/num_samples': float(count), f'{root}/uar': uar, f'{root}/mf1': mf1, f'{root}/score': score, f'{root}/tn': float(tn), f'{root}/fp': float(fp), f'{root}/fn': float(fn), f'{root}/tp': float(tp)})
        scores.append(score)
    output[f'{prefix}/mean_score'] = float(sum(scores) / len(scores))
    return output


@dataclass
class WSMSegmentMetricsCallback(BaseCallback):
    splits: list[str] = field(default_factory=lambda: ['auto'])
    task: str = 'depression'
    task_names: list[str] = field(default_factory=lambda: ['depression', 'parkinson'])
    class_names: list[str] = field(default_factory=lambda: ['healthy', 'pathological'])
    artifact_path: str = 'figures/wsm_segment_confusion_matrices'
    filename_template: str = 'epoch_{epoch}.pdf'

    def on_fit_start(self, trainer: Any) -> None:
        trainer.config.collect_cache = True

    @torch.no_grad()
    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        native_metrics: dict[str, float] = {}
        cached_by_split: dict[str, CachedSplitOutputs] = {}
        panels: list[dict[str, Any]] = []
        for split in self._resolve_cached_splits(trainer):
            cached = trainer.get_cached_split_outputs(split)
            if cached is None:
                continue
            cached_by_split[split] = cached
            native_metrics.update(self._cached_sparse_metrics(split, cached))
        legacy_metrics = self._multitask_metrics(logs)
        logs.update(legacy_metrics)
        logs.update(native_metrics)
        callback_metrics = {**legacy_metrics, **native_metrics}
        if trainer.mlflow_logger is not None and callback_metrics:
            trainer.mlflow_logger.log_metrics(callback_metrics, step=epoch)
        for split, cached in cached_by_split.items():
            if self._cached_sparse_metrics(split, cached):
                continue
            panel = self._split_panel(split, cached, logs)
            if panel is not None:
                panels.append(panel)
        if trainer.mlflow_logger is not None and panels:
            self._log_image(trainer.mlflow_logger, panels, epoch)
        if callback_metrics:
            shown = " | ".join(f"{key}={value:.4f}" for key, value in sorted(callback_metrics.items()) if "mean_macro" in key or key.endswith("/mean_score"))
            self._info(trainer, f"[WSMSegmentMetricsCallback] {shown}")

    def _resolve_cached_splits(self, trainer: Any) -> list[str]:
        if 'auto' in set(self.splits):
            return sorted(getattr(trainer, 'cached_outputs', {}).keys())
        return [split for split, _ in resolve_splits(trainer, self.splits)]

    def _cached_sparse_metrics(self, split: str, cached: CachedSplitOutputs) -> dict[str, float]:
        logits = CachedSplitOutputs._concat_chunks(cached.preds)
        targets = CachedSplitOutputs._concat_chunks(cached.targets)
        if logits is None or targets is None or logits.ndim != 2 or targets.ndim != 2:
            return {}
        if tuple(logits.shape[1:]) != (2,) or tuple(targets.shape) != tuple(logits.shape):
            return {}
        observed_mask = self._cached_observed_mask(cached, int(logits.shape[0]))
        prefix = {'val': 'dev', 'test': 'test_none'}.get(split, split)
        if prefix not in {'dev', 'test_none', 'test_soft', 'test_hard'}:
            return {}
        return compute_sparse_two_task_metrics(logits, targets, observed_mask, prefix, tuple(self.task_names))

    @staticmethod
    def _concat_value(value: Any) -> torch.Tensor | None:
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, list) and value and all(torch.is_tensor(item) for item in value):
            return torch.cat([item.detach().cpu() for item in value], dim=0)
        return None

    def _cached_observed_mask(self, cached: CachedSplitOutputs, count: int) -> torch.Tensor:
        for attribute in ('observed_mask', 'masks'):
            raw_value = getattr(cached, attribute, None)
            if isinstance(raw_value, dict):
                raw_value = raw_value.get("observed_mask")
            value = self._concat_value(raw_value)
            if value is not None:
                if tuple(value.shape) != (count, 2):
                    raise ValueError('cached observed_mask must have shape [N,2]')
                return value.to(dtype=torch.bool)
        metadata = getattr(cached, 'sample_meta', None)
        if isinstance(metadata, list) and len(metadata) >= count:
            masks: list[torch.Tensor] = []
            for item in metadata[:count]:
                if not isinstance(item, dict):
                    break
                value = item.get('observed_mask')
                if value is not None:
                    masks.append(torch.as_tensor(value, dtype=torch.bool).reshape(2))
                elif 'observed_depression' in item and 'observed_parkinson' in item:
                    masks.append(torch.tensor([item['observed_depression'], item['observed_parkinson']], dtype=torch.bool))
                elif item.get('corpus') == 'depression':
                    masks.append(torch.tensor([True, False]))
                elif item.get('corpus') == 'parkinson':
                    masks.append(torch.tensor([False, True]))
                else:
                    break
            if len(masks) == count:
                return torch.stack(masks)
        raise ValueError('native sparse cached outputs do not provide observed_mask or ownership metadata')

    def _multitask_metrics(self, logs: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        prefixes = ('dev', 'test_none', 'test_soft', 'test_hard')
        metric_names = ('num_samples', 'loss', 'macro_precision', 'macro_recall', 'macro_f1')
        for prefix in prefixes:
            values: dict[str, list[float]] = {'loss': [], 'macro_precision': [], 'macro_recall': [], 'macro_f1': [], 'score': []}
            for task in self.task_names:
                task_metrics: dict[str, float] = {}
                for metric_name in metric_names:
                    value = logs.get(f'{prefix}_{task}/{metric_name}')
                    if value is None:
                        continue
                    value = float(value)
                    out[f'{prefix}/{task}/{metric_name}'] = value
                    task_metrics[metric_name] = value
                    if metric_name in values:
                        values[metric_name].append(value)
                if 'macro_recall' in task_metrics and 'macro_f1' in task_metrics:
                    score = 0.5 * (task_metrics['macro_recall'] + task_metrics['macro_f1'])
                    out[f'{prefix}/{task}/score'] = score
                    values['score'].append(score)
            for metric_name, items in values.items():
                if items:
                    out[f'{prefix}/mean_{metric_name}'] = float(np.mean(items))
        return out

    def _split_panel(self, split: str, cached: CachedSplitOutputs, logs: dict[str, float]) -> dict[str, Any] | None:
        logits = CachedSplitOutputs._concat_chunks(cached.preds)
        targets = CachedSplitOutputs._concat_chunks(cached.targets)
        if logits is None or targets is None or (logits.ndim == 2 and tuple(logits.shape[1:]) == (2,) and tuple(targets.shape) == tuple(logits.shape)):
            return None
        return self._panel(split, self._confusion_matrix(logits, targets), float(logs.get(f'{split}/macro_recall', float('nan'))), float(logs.get(f'{split}/macro_f1', float('nan'))))

    def _confusion_matrix(self, logits: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        output = ModelOutput(preds=logits)
        batch = Batch(inputs={}, targets=targets, meta={})
        cm = ConfusionMatrixMetric()
        cm.reset()
        cm.update(output, batch)
        cm.compute()
        return cm.value()

    def _panel(self, name: str, cm: np.ndarray, uar: float, f1: float) -> dict[str, Any]:
        return {'name': name, 'uar': uar, 'f1': f1, 'cm': cm}

    def _log_image(self, logger: Any, panels: list[dict[str, Any]], epoch: int) -> None:
        import matplotlib.pyplot as plt
        cols = min(3, max(1, len(panels)))
        rows = int(np.ceil(len(panels) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.2 * rows), squeeze=False)
        for index, ax in enumerate(axes.flatten()):
            if index >= len(panels):
                ax.axis('off')
                continue
            self._draw_cm(ax, panels[index])
        fig.suptitle(f'WSM {self.task} segment confusion matrices, epoch {epoch}', fontsize=14)
        fig.tight_layout()
        logger.log_artifact_bytes(self._fig_to_pdf_bytes(fig), artifact_path=self.artifact_path, filename=self.filename_template.format(epoch=epoch))
        plt.close(fig)

    def _draw_cm(self, ax: Any, panel: dict[str, Any]) -> None:
        cm = panel['cm']
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=np.float64), where=row_sum != 0.0) * 100.0
        ax.imshow(cm_pct, cmap='Blues', vmin=0.0, vmax=100.0)
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                color = 'white' if cm_pct[i, j] > 50.0 else 'black'
                ax.text(j, i, f"{int(cm[i, j])} {cm_pct[i, j]:.1f}%", ha="center", va="center", color=color)
        ax.set(xticks=np.arange(len(self.class_names)), yticks=np.arange(len(self.class_names)), xticklabels=self.class_names, yticklabels=self.class_names, xlabel="Predicted", ylabel="True", title="{} UAR={:.4f}, F1={:.4f}".format(panel["name"], panel["uar"], panel["f1"]))

    @staticmethod
    def _fig_to_pdf_bytes(fig: Any) -> bytes:
        from io import BytesIO
        buffer = BytesIO()
        fig.savefig(buffer, format='pdf', bbox_inches='tight')
        return buffer.getvalue()


@CALLBACKS.register('wsm_segment_metrics_callback')
def wsm_segment_metrics_callback(context: Any | None = None, **params: Any) -> WSMSegmentMetricsCallback:
    params = dict(params)
    if context is not None:
        params.setdefault('task', context.get('data.task', 'depression'))
        params.setdefault('task_names', list(context.get('data.task_names', ['depression', 'parkinson'])))
        params.setdefault('class_names', list(context.get('data.class_names', ['healthy', 'pathological'])))
    return WSMSegmentMetricsCallback(**params)


@CALLBACKS.register('wsm_audio_metrics_callback')
def wsm_audio_metrics_callback(context: Any | None = None, **params: Any) -> WSMSegmentMetricsCallback:
    return wsm_segment_metrics_callback(context=context, **params)
