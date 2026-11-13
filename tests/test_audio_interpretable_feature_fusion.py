from __future__ import annotations

import pytest

from scripts.audio_interpretable_feature_fusion.permutation import (
    feature_offsets,
    select_validation_loaders,
)


def test_permutation_selects_only_validation_loaders() -> None:
    loaders = {
        "val_care": object(),
        "val_wsm": object(),
        "test_care": object(),
        "test_wsm": object(),
    }
    selected = select_validation_loaders(loaders)
    assert set(selected) == {"val_care", "val_wsm"}
    assert not (set(selected) & {"test_care", "test_wsm"})


def test_feature_offsets_reject_incomplete_model_input() -> None:
    with pytest.raises(ValueError, match="cover the model input"):
        feature_offsets(
            {"feature_family": ["a", "b"]},
            {"family_dims": [2, 2], "input_dim": 3},
        )


def test_feature_offsets_are_contiguous_for_bepa_dimensions() -> None:
    offsets = feature_offsets(
        {"feature_family": ["a", "b", "c", "d", "e", "f", "g"]},
        {"family_dims": [1024, 3, 1024, 1024, 88, 28, 488], "input_dim": 3679},
    )
    assert [start for _, start, _ in offsets] == [0, 1024, 1027, 2051, 3075, 3163, 3191]
