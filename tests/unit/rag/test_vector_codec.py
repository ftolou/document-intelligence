from __future__ import annotations

import pytest

from receipt_intelligence.rag.vector_codec import blob_to_vector, vector_to_blob


def test_vector_blob_round_trip() -> None:
    blob = vector_to_blob([0.25, -0.5, 1.0])
    assert len(blob) == 12
    assert blob_to_vector(blob, dimension=3) == pytest.approx([0.25, -0.5, 1.0])


def test_vector_codec_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="empty"):
        vector_to_blob([])
    with pytest.raises(ValueError, match="non-finite"):
        vector_to_blob([float("nan")])
    with pytest.raises(ValueError, match="expected"):
        blob_to_vector(b"1234", dimension=2)
