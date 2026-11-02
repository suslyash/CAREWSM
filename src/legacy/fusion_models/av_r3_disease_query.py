"""Frozen availability-aware disease-query audio+video model (R3)."""
from __future__ import annotations
from typing import Any
import torch
from torch import nn
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel

class WSMAVR3DiseaseQueryModel(BaseModel):
    """Two independent learned disease queries with masked modality gates."""
    def __init__(self, audio_feature_dim: int = 768, video_feature_dim: int = 512,
                 hidden_dim: int = 192, gate_hidden_dim: int = 192,
                 dropout: float = 0.2, num_tasks: int = 2,
                 task_aware_fusion: bool = True) -> None:
        super().__init__()
        if min(audio_feature_dim, video_feature_dim, hidden_dim, gate_hidden_dim) <= 0:
            raise ValueError("R3 dimensions must be positive")
        if num_tasks != 2:
            raise ValueError("wsm_av_r3_disease_query_model requires num_tasks=2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if not isinstance(task_aware_fusion, bool):
            raise ValueError("task_aware_fusion must be boolean")
        self.audio_feature_dim = int(audio_feature_dim)
        self.video_feature_dim = int(video_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.num_tasks = int(num_tasks)
        self.task_aware_fusion = task_aware_fusion
        self.audio_projection = nn.Sequential(
            nn.LayerNorm(self.audio_feature_dim), nn.Linear(self.audio_feature_dim, self.hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.video_projection = nn.Sequential(
            nn.LayerNorm(self.video_feature_dim), nn.Linear(self.video_feature_dim, self.hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.task_queries = nn.Parameter(torch.empty(self.num_tasks, self.hidden_dim))
        nn.init.normal_(self.task_queries, mean=0.0, std=0.02)
        self.task_candidate_norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_tasks)])
        self.shared_query_gate = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim), nn.Linear(2 * self.hidden_dim, self.gate_hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(self.gate_hidden_dim, 1))
        self.task_fusion_norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_tasks)])
        self.main_heads = nn.ModuleList([self._main_head(dropout) for _ in range(self.num_tasks)])
        self.audio_aux_heads = nn.ModuleList([self._aux_head() for _ in range(self.num_tasks)])
        self.video_aux_heads = nn.ModuleList([self._aux_head() for _ in range(self.num_tasks)])

    def _main_head(self, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(self.hidden_dim, 1))

    def _aux_head(self) -> nn.Sequential:
        return nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, 1))

    def forward(self, batch: Any) -> ModelOutput:
        audio_cls = batch.inputs["audio_cls"]
        video = batch.inputs["video"]
        video_mask = batch.get_masks("video_mask")
        available = batch.get_masks("modality_available")
        if audio_cls.ndim != 2 or tuple(audio_cls.shape[1:]) != (self.audio_feature_dim,):
            raise ValueError(f"audio_cls must have shape [B,{self.audio_feature_dim}]")
        if video.ndim != 3 or video.shape[2] != self.video_feature_dim:
            raise ValueError(f"video must have shape [B,T,{self.video_feature_dim}]")
        if not torch.is_floating_point(audio_cls) or not torch.isfinite(audio_cls).all():
            raise ValueError("audio_cls must be finite floating point")
        if not torch.is_floating_point(video) or not torch.isfinite(video).all():
            raise ValueError("video must be finite floating point")
        if video_mask.dtype is not torch.bool or video_mask.shape != video.shape[:2]:
            raise ValueError("video_mask must be bool [B,T]")
        if available.dtype is not torch.bool or available.shape != (audio_cls.shape[0], 2):
            raise ValueError("modality_available must be bool [B,2]")
        if bool((~available.any(dim=1)).any()):
            raise ValueError("every sample must have at least one available modality")
        if bool((available[:, 1] & ~video_mask.any(dim=1)).any()):
            raise ValueError("available video samples need a valid video position")
        audio_available = available[:, 0:1].to(device=audio_cls.device, dtype=audio_cls.dtype)
        video_available = available[:, 1:2].to(device=video.device, dtype=video.dtype)
        effective_audio_input = audio_cls * audio_available
        video_weights = video_mask.unsqueeze(-1).to(dtype=video.dtype)
        pooled_video = (video * video_weights).sum(dim=1) / video_weights.sum(dim=1).clamp_min(1)
        effective_video_input = pooled_video * video_available
        features_audio = self.audio_projection(effective_audio_input) * audio_available
        features_video = self.video_projection(effective_video_input) * video_available
        task_audio_features, task_video_features, task_features = [], [], []
        task_weights, audio_aux_logits, video_aux_logits = [], [], []
        if self.task_aware_fusion:
            for task, query in enumerate(self.task_queries):
                query_batch = query.unsqueeze(0).expand(audio_cls.shape[0], -1)
                audio_task = self.task_candidate_norms[task](features_audio + query_batch)
                video_task = self.task_candidate_norms[task](features_video + query_batch)
                audio_score = self.shared_query_gate(torch.cat([query_batch, audio_task], dim=1)).squeeze(1)
                video_score = self.shared_query_gate(torch.cat([query_batch, video_task], dim=1)).squeeze(1)
                scores = torch.stack([audio_score, video_score], dim=1)
                scores = scores.masked_fill(~available, torch.finfo(scores.dtype).min)
                weights = torch.softmax(scores, dim=1)
                fused = self.task_fusion_norms[task](
                    query_batch + weights[:, 0:1] * audio_task + weights[:, 1:2] * video_task)
                task_audio_features.append(audio_task)
                task_video_features.append(video_task)
                task_features.append(fused)
                task_weights.append(weights)
                audio_aux_logits.append(self.audio_aux_heads[task](audio_task))
                video_aux_logits.append(self.video_aux_heads[task](video_task))
        else:
            q_shared = self.task_queries.mean(dim=0)
            query_batch = q_shared.unsqueeze(0).expand(audio_cls.shape[0], -1)
            audio_task = 0.5 * (
                self.task_candidate_norms[0](features_audio + query_batch)
                + self.task_candidate_norms[1](features_audio + query_batch))
            video_task = 0.5 * (
                self.task_candidate_norms[0](features_video + query_batch)
                + self.task_candidate_norms[1](features_video + query_batch))
            audio_score = self.shared_query_gate(torch.cat([query_batch, audio_task], dim=1)).squeeze(1)
            video_score = self.shared_query_gate(torch.cat([query_batch, video_task], dim=1)).squeeze(1)
            scores = torch.stack([audio_score, video_score], dim=1)
            scores = scores.masked_fill(~available, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1)
            fused_pre = query_batch + weights[:, 0:1] * audio_task + weights[:, 1:2] * video_task
            fused = 0.5 * (
                self.task_fusion_norms[0](fused_pre)
                + self.task_fusion_norms[1](fused_pre))
            task_audio_features.extend([audio_task, audio_task])
            task_video_features.extend([video_task, video_task])
            task_features.extend([fused, fused])
            task_weights.extend([weights, weights])
            audio_aux_logits.extend([
                self.audio_aux_heads[0](audio_task), self.audio_aux_heads[1](audio_task)])
            video_aux_logits.extend([
                self.video_aux_heads[0](video_task), self.video_aux_heads[1](video_task)])
        task_audio = torch.stack(task_audio_features, dim=1)
        task_video = torch.stack(task_video_features, dim=1)
        task_fused = torch.stack(task_features, dim=1)
        weights = torch.stack(task_weights, dim=1)
        preds = torch.cat([self.main_heads[task](task_fused[:, task])
                            for task in range(self.num_tasks)], dim=1)
        return ModelOutput(preds=preds, aux={
            "features_audio": features_audio, "features_video": features_video,
            "task_audio_features": task_audio, "task_video_features": task_video,
            "task_modality_weights": weights, "task_features": task_fused,
            "audio_aux_logits": torch.cat(audio_aux_logits, dim=1),
            "video_aux_logits": torch.cat(video_aux_logits, dim=1),
            "audio_aux_valid": available[:, 0:1].expand(-1, self.num_tasks),
            "video_aux_valid": available[:, 1:2].expand(-1, self.num_tasks),
            "task_logits": {"depression": preds[:, 0], "parkinson": preds[:, 1]},
        })

@MODELS.register("wsm_av_r3_disease_query_model")
def wsm_av_r3_disease_query_model(context: Any | None = None, **params: Any) -> WSMAVR3DiseaseQueryModel:
    params = dict(params)
    if context is not None:
        params.setdefault("audio_feature_dim", context.get("data.audio_feature_dim", 768))
        params.setdefault("video_feature_dim", context.get("data.video_feature_dim", 512))
        params.setdefault("num_tasks", context.get("data.num_tasks", 2))
    return WSMAVR3DiseaseQueryModel(**params)
