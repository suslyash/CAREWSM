from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import DATAMODULES
from chimera_ml.data.datamodule import DataModule
from common.data.wsm_manifest import build_manifest
from common.utils.segment_index import build_wsm_multitask_segment_index
from video.features.clip_video_features import PREPROCESSING_VERSION, ROI_POLICY_VERSION

MODEL_NAME = 'openai/clip-vit-base-patch32'
MODEL_REVISION = 'b97b0100e55e367c057773c2a614676470b0d575'
DEFAULT_CACHE_ROOT = '/media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache'
YOLO_SHA256 = 'a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43'
TARGET_FRAMES = 60
FEATURE_DIM = 512
TASK_NAMES = ('depression', 'parkinson')


class VideoCacheDataError(ValueError):
    pass


def _filter_flag(value: Any, field: str) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        value = value.strip()
        if value == '' or value.casefold() == 'nan':
            return False
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise VideoCacheDataError(f'{field} must be missing, 0, or 1: {value!r}') from exc
    if math.isnan(numeric):
        return False
    if numeric not in (0.0, 1.0):
        raise VideoCacheDataError(f'{field} must be missing, 0, or 1: {value!r}')
    return numeric == 1.0


def _load_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise VideoCacheDataError(f'cache index is missing: {path}')
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VideoCacheDataError(f'invalid cache index JSON at line {line_number}') from exc
            segment_id = record.get('segment_id')
            if not isinstance(segment_id, str) or not segment_id:
                raise VideoCacheDataError(f'cache index line {line_number} has no segment_id')
            if segment_id in records:
                raise VideoCacheDataError(f'duplicate cache index segment_id: {segment_id}')
            records[segment_id] = record
    return records


def _validate_artifact(record: dict[str, Any], row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = Path(record['cache_path'])
    if not path.is_file():
        raise VideoCacheDataError(f'cache artifact is missing: {path}')
    artifact = torch.load(path, map_location='cpu', weights_only=False)
    if artifact.get('segment_id') != row['segment_id'] or artifact.get('segment_id') != record['segment_id']:
        raise VideoCacheDataError(f'segment_id mismatch for {row['segment_id']}')
    if artifact.get('cache_fingerprint') != record.get('cache_fingerprint'):
        raise VideoCacheDataError(f'cache fingerprint mismatch for {row['segment_id']}')
    if artifact.get('model_name') != MODEL_NAME or artifact.get('model_revision') != MODEL_REVISION:
        raise VideoCacheDataError(f'CLIP identity mismatch for {row['segment_id']}')
    if artifact.get('detector_weights_sha256') != YOLO_SHA256:
        raise VideoCacheDataError(f'YOLO SHA mismatch for {row['segment_id']}')
    if artifact.get('detection_parameters') != {'confidence': 0.5, 'iou': 0.5, 'imgsz': 640}:
        raise VideoCacheDataError(f'detector settings mismatch for {row['segment_id']}')
    preprocessing = artifact.get('preprocessing')
    if not isinstance(preprocessing, dict) or preprocessing.get('target_frames') != TARGET_FRAMES:
        raise VideoCacheDataError(f'target_frames metadata mismatch for {row['segment_id']}')
    if preprocessing.get('version') != PREPROCESSING_VERSION:
        raise VideoCacheDataError(f'preprocessing version mismatch for {row['segment_id']}')
    if preprocessing.get('roi_policy_version') != ROI_POLICY_VERSION:
        raise VideoCacheDataError(f'ROI policy version mismatch for {row['segment_id']}')
    features = artifact.get('features')
    valid_mask = artifact.get('valid_mask')
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise VideoCacheDataError(f'features must be [T,{FEATURE_DIM}] for {row['segment_id']}')
    temporal_length = int(features.shape[0])
    if not 1 <= temporal_length <= TARGET_FRAMES:
        raise VideoCacheDataError(f'invalid temporal length for {row['segment_id']}: {temporal_length}')
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool or tuple(valid_mask.shape) != (temporal_length,):
        raise VideoCacheDataError(f'valid_mask mismatch for {row['segment_id']}')
    if not bool(valid_mask.all()):
        raise VideoCacheDataError(f'valid_mask is not all true for {row['segment_id']}')
    if not bool(torch.isfinite(features).all()):
        raise VideoCacheDataError(f'features are not finite for {row['segment_id']}')
    sources = artifact.get('frame_sources')
    if not isinstance(sources, list) or len(sources) != temporal_length:
        raise VideoCacheDataError(f'frame_sources mismatch for {row['segment_id']}')
    if any(source not in {'body_roi', 'full_frame_fallback'} for source in sources):
        raise VideoCacheDataError(f'invalid frame source for {row['segment_id']}')
    detected = int(artifact.get('detected_body_count', -1))
    fallback = int(artifact.get('full_frame_fallback_count', -1))
    if detected + fallback != temporal_length or detected != sources.count('body_roi') or fallback != sources.count('full_frame_fallback'):
        raise VideoCacheDataError(f'detection/fallback provenance mismatch for {row['segment_id']}')
    if float(artifact.get('detection_coverage', -1.0)) != detected / temporal_length:
        raise VideoCacheDataError(f'detection coverage mismatch for {row['segment_id']}')
    if float(artifact.get('fallback_coverage', -1.0)) != fallback / temporal_length:
        raise VideoCacheDataError(f'fallback coverage mismatch for {row['segment_id']}')
    sampled_indices = artifact.get('sampled_frame_indices')
    if not isinstance(sampled_indices, list) or len(sampled_indices) != temporal_length:
        raise VideoCacheDataError(f'sampled indices mismatch for {row['segment_id']}')
    if any(not isinstance(index, int) or index < 0 for index in sampled_indices):
        raise VideoCacheDataError(f'invalid sampled indices for {row['segment_id']}')
    if any(left > right for left, right in zip(sampled_indices, sampled_indices[1:])):
        raise VideoCacheDataError(f'sampled indices are not chronological for {row['segment_id']}')
    if not bool(torch.isfinite(features[valid_mask]).all()):
        raise VideoCacheDataError(f'valid features are not finite for {row['segment_id']}')
    if not torch.equal(features[~valid_mask], torch.zeros_like(features[~valid_mask])):
        raise VideoCacheDataError(f'invalid features are not exact zero for {row['segment_id']}')
    metadata = {
        'segment_id': row['segment_id'],
        'video_id': row['video_id'],
        'corpus': row['corpus'],
        'split': row['split'],
        'cache_path': str(path),
        'cache_fingerprint': record['cache_fingerprint'],
        'temporal_length': temporal_length,
        'detected_body_count': int(record.get('detected_body_count', artifact['detected_body_count'])),
        'full_frame_fallback_count': int(record.get('full_frame_fallback_count', artifact['full_frame_fallback_count'])),
        'detection_coverage': float(record.get('detection_coverage', artifact['detection_coverage'])),
        'fallback_coverage': float(record.get('fallback_coverage', artifact['fallback_coverage'])),
        'frame_sources': list(artifact['frame_sources']),
    }
    return features.float(), valid_mask.clone(), metadata


def _row_targets(row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    values = [row['y_depression'], row['y_parkinson']]
    targets = torch.tensor([float(value) if value is not None else float('nan') for value in values], dtype=torch.float32)
    observed = torch.tensor([bool(row['observed_depression']), bool(row['observed_parkinson'])], dtype=torch.bool)
    if any(observed[index] != (values[index] is not None) for index in range(2)):
        raise VideoCacheDataError(f'target/observed mismatch for {row['segment_id']}')
    return targets, observed


class WSMVideoCacheDataset(Dataset):
    def __init__(self, samples: Iterable[dict[str, Any]]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index]
        return {
            'inputs': {'video': sample['features'].clone()},
            'targets': sample['targets'].clone(),
            'observed_mask': sample['observed_mask'].clone(),
            'video_mask': sample['video_mask'].clone(),
            'meta': dict(sample['meta']),
        }


def collate_wsm_video_cache(samples: list[dict[str, Any]]) -> Batch:
    if not samples:
        raise VideoCacheDataError('cannot collate an empty video cache batch')
    lengths = [int(sample['inputs']['video'].shape[0]) for sample in samples]
    batch_max_t = max(lengths)
    padded_video = torch.zeros(len(samples), batch_max_t, FEATURE_DIM, dtype=torch.float32)
    video_mask = torch.zeros(len(samples), batch_max_t, dtype=torch.bool)
    for batch_index, (sample, length) in enumerate(zip(samples, lengths, strict=True)):
        features = sample['inputs']['video'].float()
        artifact_mask = sample['video_mask'].to(dtype=torch.bool)
        if tuple(features.shape) != (length, FEATURE_DIM) or tuple(artifact_mask.shape) != (length,):
            raise VideoCacheDataError('sample shape does not match its temporal length')
        padded_video[batch_index, :length] = features
        video_mask[batch_index, :length] = artifact_mask
    return Batch(
        inputs={'video': padded_video},
        targets=torch.stack([sample['targets'].float() for sample in samples]),
        masks={
            'video_mask': video_mask,
            'observed_mask': torch.stack([sample['observed_mask'].bool() for sample in samples]),
        },
        meta={'sample_meta': [sample['meta'] for sample in samples]},
    )


class WSMVideoCacheDataModule(DataModule):
    def __init__(self, data_root: str = '/media/maxim/Databases/WSM_NEW', cache_root: str = DEFAULT_CACHE_ROOT, cache_index_path: str | None = None, batch_size: int = 32, num_workers: int = 0, pin_memory: bool = True, persistent_workers: bool = False, shuffle_train: bool = True, drop_last_train: bool = False) -> None:
        self.data_root = str(Path(data_root).expanduser().resolve())
        self.cache_root = str(Path(cache_root).expanduser().resolve())
        self.cache_index_path = str(Path(cache_index_path).expanduser().resolve()) if cache_index_path else str(Path(self.cache_root) / 'cache_index.jsonl')
        canonical_rows, self.manifest_audit = build_manifest(self.data_root)
        rows = [row for row in canonical_rows if row['split'] in {'train', 'dev'}]
        test_rows = [row for row in canonical_rows if row['split'] == 'test']
        raw_records = build_wsm_multitask_segment_index(self.data_root)
        raw_test = raw_records[raw_records['split'] == 'test']
        raw_membership: dict[tuple[str, str, str], tuple[bool, bool]] = {}
        for raw in raw_test.itertuples(index=False):
            key = (str(raw.task), str(raw.video_id), str(raw.segment_file))
            if key in raw_membership:
                raise VideoCacheDataError(f'duplicate raw Test metadata for {key}')
            raw_membership[key] = (
                _filter_flag(raw.soft_filter, 'soft_filter'),
                _filter_flag(raw.hard_filter, 'hard_filter'),
            )
        if len(raw_membership) != len(test_rows):
            raise VideoCacheDataError(f'raw Test membership count {len(raw_membership)} does not match canonical count {len(test_rows)}')
        test_protocols = ('test_none', 'test_soft', 'test_hard')
        test_membership: dict[str, list[dict[str, Any]]] = {protocol: [] for protocol in test_protocols}
        for row in test_rows:
            corpus, video_id, segment_file = json.loads(row['segment_id'])
            membership = raw_membership.get((corpus, video_id, segment_file))
            if membership is None:
                raise VideoCacheDataError(f"missing raw Test metadata for {row['segment_id']}")
            test_membership['test_none'].append(row)
            if membership[0]:
                test_membership['test_soft'].append(row)
            if membership[1]:
                test_membership['test_hard'].append(row)
        self.raw_test_protocol_counts = {protocol: len(protocol_rows) for protocol, protocol_rows in test_membership.items()}
        index = _load_index(Path(self.cache_index_path))
        self.train_unavailable_count = 0
        self.dev_unavailable_count = 0
        train_samples: list[dict[str, Any]] = []
        dev_samples: list[dict[str, Any]] = []
        for row in rows:
            segment_id = row['segment_id']
            record = index.get(segment_id)
            if record is None:
                raise VideoCacheDataError(f'missing cache index row for {segment_id}')
            if record.get('split') != row['split']:
                raise VideoCacheDataError(f'cache split mismatch for {segment_id}')
            status = record.get('status')
            if status == 'failed':
                if record.get('failure_category') != 'no_body_detected':
                    raise VideoCacheDataError(f"unexpected cache failure for {segment_id}: {record.get('failure_category')}")
                if row['split'] == 'train':
                    self.train_unavailable_count += 1
                else:
                    self.dev_unavailable_count += 1
                continue
            if status not in {'extracted', 'reused'}:
                raise VideoCacheDataError(f'unexpected cache status for {segment_id}: {status}')
            features, valid_mask, metadata = _validate_artifact(record, row)
            targets, observed_mask = _row_targets(row)
            sample = {'features': features, 'video_mask': valid_mask, 'targets': targets, 'observed_mask': observed_mask, 'meta': metadata}
            (train_samples if row['split'] == 'train' else dev_samples).append(sample)
        test_samples: dict[str, list[dict[str, Any]]] = {protocol: [] for protocol in test_protocols}
        self.test_unavailable_count = {protocol: 0 for protocol in test_protocols}
        for row in test_rows:
            segment_id = row['segment_id']
            record = index.get(segment_id)
            if record is None:
                raise VideoCacheDataError(f'missing cache index row for {segment_id}')
            if record.get('split') != 'test':
                raise VideoCacheDataError(f'cache split mismatch for {segment_id}')
            protocols = [protocol for protocol in test_protocols if row in test_membership[protocol]]
            status = record.get('status')
            if status == 'failed':
                if record.get('failure_category') != 'no_body_detected':
                    raise VideoCacheDataError(f"unexpected Test cache failure for {segment_id}: {record.get('failure_category')}")
                for protocol in protocols:
                    self.test_unavailable_count[protocol] += 1
                continue
            if status not in {'extracted', 'reused'}:
                raise VideoCacheDataError(f'unexpected Test cache status for {segment_id}: {status}')
            features, valid_mask, metadata = _validate_artifact(record, row)
            targets, observed_mask = _row_targets(row)
            for protocol in protocols:
                protocol_meta = dict(metadata)
                protocol_meta['evaluation_protocol'] = protocol
                test_samples[protocol].append({'features': features, 'video_mask': valid_mask, 'targets': targets, 'observed_mask': observed_mask, 'meta': protocol_meta})
        self.train_dataset = WSMVideoCacheDataset(train_samples)
        self.val_dataset = WSMVideoCacheDataset(dev_samples)
        self.test_dataset = {protocol: WSMVideoCacheDataset(test_samples[protocol]) for protocol in test_protocols}
        self.test_rows_indexed = sum(record.get('split') == 'test' for record in index.values())
        self.test_rows_loaded = sum(len(dataset) for dataset in self.test_dataset.values())
        self.collate_fn = collate_wsm_video_cache
        self.video_cache_success_total = len(train_samples) + len(dev_samples) + len(test_samples['test_none'])
        self.video_cache_failure_total = self.train_unavailable_count + self.dev_unavailable_count + self.test_unavailable_count['test_none']
        super().__init__(train_dataset=self.train_dataset, val_dataset=self.val_dataset, test_dataset=self.test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, shuffle_train=shuffle_train, drop_last_train=drop_last_train, collate_fn=self.collate_fn)

    def val_dataloader(self) -> dict[str, Any]:
        return {
            'dev': self._make_loader(self.val_dataset, shuffle=False, drop_last=False),
            'test_none': self._make_loader(self.test_dataset['test_none'], shuffle=False, drop_last=False),
            'test_soft': self._make_loader(self.test_dataset['test_soft'], shuffle=False, drop_last=False),
            'test_hard': self._make_loader(self.test_dataset['test_hard'], shuffle=False, drop_last=False),
        }

    def describe_context(self, context: Any) -> None:
        context.set('data.num_tasks', 2)
        context.set('data.task_names', list(TASK_NAMES))
        context.set('data.video_feature_dim', FEATURE_DIM)
        context.set('data.video_sequence_steps', TARGET_FRAMES)
        context.set('data.video_cache_root', self.cache_root)
        context.set('data.video_train_rows', len(self.train_dataset))
        context.set('data.video_dev_rows', len(self.val_dataset))
        context.set('data.video_train_unavailable', self.train_unavailable_count)
        context.set('data.video_dev_unavailable', self.dev_unavailable_count)
        context.set('data.video_cache_success_total', self.video_cache_success_total)
        context.set('data.video_cache_failure_total', self.video_cache_failure_total)
        context.set('data.video_cache_variable_length', True)
        context.set('data.test_protocols', ['test_none', 'test_soft', 'test_hard'])
        context.set('data.video_test_none_rows', len(self.test_dataset['test_none']))
        context.set('data.video_test_soft_rows', len(self.test_dataset['test_soft']))
        context.set('data.video_test_hard_rows', len(self.test_dataset['test_hard']))
        context.set('data.video_test_none_unavailable', self.test_unavailable_count['test_none'])
        context.set('data.video_test_soft_unavailable', self.test_unavailable_count['test_soft'])
        context.set('data.video_test_hard_unavailable', self.test_unavailable_count['test_hard'])
        context.set('data.video_test_rows_indexed', self.test_rows_indexed)
        context.set('data.video_epoch_test_monitoring_required', True)
        context.set('data.test_rows_loaded', self.test_rows_loaded)


@DATAMODULES.register('wsm_video_depart_v1_datamodule')
def wsm_video_depart_v1_datamodule(context: Any | None = None, **params: Any) -> WSMVideoCacheDataModule:
    del context
    return WSMVideoCacheDataModule(**params)
