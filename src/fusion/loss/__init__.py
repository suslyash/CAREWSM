"""Fusion loss utilities."""
from .ramps_r1_teacher import (
    binary_brier,
    binary_ece,
    binary_nll,
    fit_binary_temperature,
    select_class_threshold,
)
__all__ = [
    "binary_brier", "binary_ece", "binary_nll",
    "fit_binary_temperature", "select_class_threshold",
]

from .ramps_r2_reliability import (apply_reliability_rule, audio_centroid_distance, class_confidence, detached_reliability, empirical_ood_percentile, normalized_audio_class_centroids, normalized_binary_entropy, same_class_agreement, select_reliability_rules, stratified_folds)

from .ramps_r2_semantic import apply_semantic_rule, fit_monotonic_semantic_calibrator, masked_pool_clip_frames, normalize_prompt_bank, search_semantic_rules, semantic_cosine_margin, semantic_hash, semantic_metrics
