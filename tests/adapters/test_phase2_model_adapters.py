from __future__ import annotations

from receipt_intelligence.adapters.multimodal.ollama import _extract_text
from receipt_intelligence.adapters.text_detection.paddle import (
    _extract_polygons_and_scores,
)


def test_ollama_multimodal_accepts_qwen_thinking_transport_field() -> None:
    text, source = _extract_text({"message": {"content": "", "thinking": "VISIBLE RECEIPT TEXT"}})

    assert text == "VISIBLE RECEIPT TEXT"
    assert source == "message.thinking"


def test_paddle_adapter_extracts_polygons_from_nested_result() -> None:
    polygons, scores = _extract_polygons_and_scores(
        {
            "result": {
                "dt_polys": [
                    [[0, 0], [10, 0], [10, 5], [0, 5]],
                    [[0, 10], [10, 10], [10, 15], [0, 15]],
                ],
                "dt_scores": [0.9, 0.8],
            }
        }
    )

    assert len(polygons) == 2
    assert scores == [0.9, 0.8]
