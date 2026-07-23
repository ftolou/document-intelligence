"""Compact float32 vector serialization for SQLite BLOB storage."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence


def vector_to_blob(vector: Sequence[float]) -> bytes:
    """Serialize one finite, non-empty vector as little-endian float32 bytes."""

    values = [float(value) for value in vector]
    if not values:
        raise ValueError("Embedding vector must not be empty.")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Embedding vector contains a non-finite value.")
    return struct.pack(f"<{len(values)}f", *values)


def blob_to_vector(blob: bytes, *, dimension: int) -> list[float]:
    """Deserialize a little-endian float32 BLOB with an expected dimension."""

    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    expected_size = dimension * 4
    if len(blob) != expected_size:
        raise ValueError(
            f"Embedding BLOB has {len(blob)} bytes, expected {expected_size} for dimension {dimension}."
        )
    return list(struct.unpack(f"<{dimension}f", blob))
