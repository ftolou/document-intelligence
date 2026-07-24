"""PaddleOCR adapter for the OCR engine port."""

from __future__ import annotations

from typing import Any

from receipt_intelligence.application.ports.ocr import OcrEngine, OcrRequest
from receipt_intelligence.engines.ocr_engine import run_paddleocr_image


class PaddleOcrEngine(OcrEngine):
    def recognize(self, request: OcrRequest) -> dict[str, Any]:
        return run_paddleocr_image(
            image_path=request.image_path,
            out_json_path=request.out_json_path,
            work_dir=request.work_dir,
            lang=request.lang,
            device=request.device,
            max_side_limit=request.max_side_length,
            use_angle_cls=request.detect_orientation,
            det_limit_side_len=request.detection_max_side_length,
            progress_callback=request.progress_callback,
        )


__all__ = ["PaddleOcrEngine"]
