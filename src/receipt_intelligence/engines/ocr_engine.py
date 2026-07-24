#!/usr/bin/env python3
"""V14 PaddleOCR wrapper using the known-working V13 Docker runtime profile.

This file intentionally avoids the experimental PaddleOCR constructor options that
were added in the first V14 package. V13 worked in Docker with PaddlePaddle 3.2.0,
PaddleOCR 3.x, PIR disabled, and MKLDNN/oneDNN disabled. This module keeps that
same OCR behavior and only converts the OCR output into the V14 JSON shape used
by the LLM-main parser.
"""

from __future__ import annotations

# IMPORTANT: these must be set before Paddle/PaddleOCR is imported anywhere.
import os

os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")

import json
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from receipt_intelligence.utils.image_utils import prepare_image_for_ocr

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _emit(
    callback: Callable[[dict[str, Any]], None] | None,
    stage: str,
    status: str,
    message: str,
    **details: Any,
) -> None:
    if callback is None:
        return
    try:
        callback({"stage": stage, "status": status, "message": message, "details": details})
    except Exception:
        pass


@lru_cache(maxsize=8)
def _get_paddleocr(
    lang: str,
    device: str,
    text_detection_model_name: str | None,
    text_recognition_model_name: str | None,
):
    """V13-compatible PaddleOCR initialization.

    Do not add engine_config/run_mode/new_ir kwargs here. The point of this V14
    package is to keep OCR identical to the Docker setup that already worked in
    V13, while replacing only the semantic receipt parser.
    """
    from paddleocr import PaddleOCR

    kwargs: dict[str, Any] = {
        "lang": lang,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    if device and device.lower() not in {"auto", "default"}:
        kwargs["device"] = "gpu:0" if device.lower() in {"cuda", "gpu"} else device
    if text_detection_model_name:
        kwargs["text_detection_model_name"] = text_detection_model_name
    if text_recognition_model_name:
        kwargs["text_recognition_model_name"] = text_recognition_model_name

    try:
        return PaddleOCR(**kwargs)
    except TypeError:
        # PaddleOCR 2.x compatibility fallback.
        legacy_kwargs: dict[str, Any] = {
            "lang": lang,
            "use_angle_cls": True,
            "use_gpu": device.lower() in {"cuda", "gpu", "gpu:0"},
            "show_log": False,
        }
        return PaddleOCR(**legacy_kwargs)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def _as_list(obj: Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


def _unwrap_ocr_result_obj(obj: Any) -> Any:
    if hasattr(obj, "json"):
        try:
            j = obj.json
            if callable(j):
                j = j()
            obj = j
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            obj = obj.to_dict()
        except Exception:
            pass
    if isinstance(obj, dict) and "res" in obj and isinstance(obj["res"], dict):
        obj = obj["res"]
    return obj


def _poly_to_bbox(poly: Any) -> tuple[int, int, int, int, list[list[int]]]:
    arr = np.asarray(poly, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.shape[-1] != 2:
        raise ValueError(f"Invalid polygon shape: {arr.shape}")
    xs = arr[:, 0]
    ys = arr[:, 1]
    xmin = int(np.floor(xs.min()))
    ymin = int(np.floor(ys.min()))
    xmax = int(np.ceil(xs.max()))
    ymax = int(np.ceil(ys.max()))
    polygon = [[int(round(x)), int(round(y))] for x, y in arr.tolist()]
    return xmin, ymin, xmax, ymax, polygon


def _box_to_bbox(box: Any) -> tuple[int, int, int, int, list[list[int]]]:
    arr = np.asarray(box, dtype=float)
    if arr.ndim == 1 and arr.size == 4:
        xmin, ymin, xmax, ymax = [int(round(v)) for v in arr.tolist()]
        polygon = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        return xmin, ymin, xmax, ymax, polygon
    return _poly_to_bbox(box)


def _normalize_v3_result(
    result_obj: Any, image_width: int, image_height: int, min_score: float
) -> list[dict[str, Any]]:
    res = _unwrap_ocr_result_obj(result_obj)
    if not isinstance(res, dict):
        return []

    texts = _as_list(res.get("rec_texts") or res.get("texts") or res.get("text"))
    scores = _as_list(res.get("rec_scores") or res.get("scores") or res.get("score"))
    polys = res.get("rec_polys") or res.get("dt_polys") or res.get("polys") or None
    boxes = res.get("rec_boxes") or res.get("dt_boxes") or res.get("boxes") or None

    polys_list = _as_list(polys) if polys is not None else []
    boxes_list = _as_list(boxes) if boxes is not None else []

    words: list[dict[str, Any]] = []
    for i, text in enumerate(texts):
        text = str(text or "").strip()
        score = float(scores[i]) if i < len(scores) and scores[i] is not None else 0.0
        if not text or score < min_score:
            continue

        shape_obj = (
            polys_list[i]
            if i < len(polys_list)
            else (boxes_list[i] if i < len(boxes_list) else None)
        )
        if shape_obj is None:
            continue
        try:
            if i < len(polys_list):
                xmin, ymin, xmax, ymax, polygon = _poly_to_bbox(shape_obj)
            else:
                xmin, ymin, xmax, ymax, polygon = _box_to_bbox(shape_obj)
        except Exception:
            continue

        width = max(0, xmax - xmin)
        height = max(0, ymax - ymin)
        if width <= 0 or height <= 0:
            continue

        words.append(
            {
                "id": f"paddle_{len(words):04d}",
                "text": text,
                "confidence": score,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "width": width,
                "height": height,
                "polygon": polygon,
                "bbox": {
                    "x": xmin / max(image_width, 1),
                    "y": ymin / max(image_height, 1),
                    "w": width / max(image_width, 1),
                    "h": height / max(image_height, 1),
                },
            }
        )
    return words


def _normalize_v2_result(
    result_obj: Any, image_width: int, image_height: int, min_score: float
) -> list[dict[str, Any]]:
    data = result_obj
    if isinstance(data, tuple):
        data = list(data)
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        if not (
            len(data[0]) == 2
            and isinstance(data[0][1], (tuple, list))
            and isinstance(data[0][1][0], str)
        ):
            data = data[0]

    words: list[dict[str, Any]] = []
    if not isinstance(data, list):
        return words

    for line in data:
        try:
            box = line[0]
            text_score = line[1]
            text = str(text_score[0] or "").strip()
            score = float(text_score[1])
        except Exception:
            continue
        if not text or score < min_score:
            continue
        try:
            xmin, ymin, xmax, ymax, polygon = _poly_to_bbox(box)
        except Exception:
            continue
        width = max(0, xmax - xmin)
        height = max(0, ymax - ymin)
        if width <= 0 or height <= 0:
            continue
        words.append(
            {
                "id": f"paddle_{len(words):04d}",
                "text": text,
                "confidence": score,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "width": width,
                "height": height,
                "polygon": polygon,
                "bbox": {
                    "x": xmin / max(image_width, 1),
                    "y": ymin / max(image_height, 1),
                    "w": width / max(image_width, 1),
                    "h": height / max(image_height, 1),
                },
            }
        )
    return words


def _run_paddle_ocr_v13_profile(
    image_path: str | Path,
    lang: str = "german",
    device: str = "cpu",
    min_score: float = 0.30,
    text_detection_model_name: str | None = None,
    text_recognition_model_name: str | None = "latin_PP-OCRv5_mobile_rec",
) -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)
    image_width, image_height = _image_size(path)

    ocr = _get_paddleocr(lang, device, text_detection_model_name, text_recognition_model_name)

    raw: Any
    api_used: str
    if hasattr(ocr, "predict"):
        api_used = "predict"
        raw = ocr.predict(str(path))
    else:
        api_used = "ocr"
        raw = ocr.ocr(str(path), cls=True)

    words: list[dict[str, Any]] = []
    if api_used == "predict":
        for obj in _as_list(raw):
            words.extend(_normalize_v3_result(obj, image_width, image_height, min_score))
        if not words:
            words = _normalize_v2_result(raw, image_width, image_height, min_score)
    else:
        words = _normalize_v2_result(raw, image_width, image_height, min_score)

    words.sort(key=lambda w: (w["ymin"], w["xmin"]))
    return {
        "image_path": str(path),
        "image_width": image_width,
        "image_height": image_height,
        "engine": "paddleocr",
        "paddle_api_used": api_used,
        "lang": lang,
        "device": device,
        "min_score": min_score,
        "text_detection_model_name": text_detection_model_name,
        "text_recognition_model_name": text_recognition_model_name,
        "word_count": len(words),
        "words": words,
    }


def _words_to_receipt_lines(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Treat each PaddleOCR text box as one V14 line.

    PaddleOCR usually returns line-level boxes, not token-level words. This is the
    same evidence granularity V13 used. The V14 LLM parser can consume it as
    coordinate-preserving line context.
    """
    lines: list[dict[str, Any]] = []
    for i, w in enumerate(
        sorted(words, key=lambda r: (float(r.get("ymin", 0)), float(r.get("xmin", 0))))
    ):
        line_id = f"line_{i:03d}"
        source_id = str(w.get("id") or f"paddle_{i:04d}")
        lines.append(
            {
                "line_id": line_id,
                "source_word_ids": [source_id],
                "text": str(w.get("text", "")).strip(),
                "confidence": float(w.get("confidence") or 0.0),
                "xmin": float(w.get("xmin") or 0.0),
                "ymin": float(w.get("ymin") or 0.0),
                "xmax": float(w.get("xmax") or 0.0),
                "ymax": float(w.get("ymax") or 0.0),
                "polygon": w.get("polygon") or [],
            }
        )
    return [ln for ln in lines if ln["text"]]


def run_paddleocr_image(
    image_path: Path,
    out_json_path: Path,
    *,
    work_dir: Path | None = None,
    lang: str = "german",
    device: str = "cpu",
    max_side_limit: int = 4000,
    use_angle_cls: bool = True,  # kept for app/API compatibility; V13 3.x profile disables textline orientation.
    det_limit_side_len: int = 4000,  # kept for app/API compatibility; V13 profile uses image pre-resize instead.
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {image_path.suffix}")
    work_dir = Path(work_dir or out_json_path.parent)

    _emit(
        progress_callback,
        "ocr_full_image",
        "running",
        "Preparing image for PaddleOCR.",
        image=str(image_path),
    )
    prepared = prepare_image_for_ocr(image_path, work_dir, max_side_limit=max_side_limit)
    if prepared["resized"]:
        _emit(
            progress_callback,
            "ocr_full_image",
            "running",
            "Image exceeded max side limit and was resized before OCR.",
            original_width=prepared["original_width"],
            original_height=prepared["original_height"],
            image_width=prepared["image_width"],
            image_height=prepared["image_height"],
            max_side_limit=max_side_limit,
        )

    _emit(
        progress_callback,
        "ocr_full_image",
        "running",
        "Initializing PaddleOCR with V13-compatible Docker profile.",
        lang=lang,
        device=device,
    )
    _emit(progress_callback, "ocr_full_image", "running", "Running PaddleOCR on the full image.")
    try:
        v13_ocr = _run_paddle_ocr_v13_profile(
            prepared["prepared_path"],
            lang=lang,
            device=device,
            min_score=0.30,
            text_detection_model_name=None,
            text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
        )
    except Exception as exc:
        msg = str(exc)
        if any(
            n in msg
            for n in [
                "ConvertPirAttribute2RuntimeAttribute",
                "onednn_instruction",
                "pir::ArrayAttribute",
                "MKLDNN",
                "oneDNN",
            ]
        ):
            raise RuntimeError(
                "PaddleOCR failed inside PaddlePaddle oneDNN/MKLDNN/PIR runtime. "
                "This V14 package is configured to match the known-working V13 Docker runtime. "
                "Rebuild the Docker image with --no-cache and confirm docker-compose.yml contains "
                "FLAGS_enable_pir_api=0, FLAGS_use_mkldnn=0, and FLAGS_use_onednn=0."
            ) from exc
        raise

    words = v13_ocr.get("words", [])
    lines = _words_to_receipt_lines(words)

    ocr_json = {
        "schema_version": "v14_app_ocr_v13_runtime_1",
        "source_image": str(image_path),
        "ocr_image": str(prepared["prepared_path"]),
        "image_width": prepared["image_width"],
        "image_height": prepared["image_height"],
        "original_width": prepared["original_width"],
        "original_height": prepared["original_height"],
        "resized_for_ocr": prepared["resized"],
        "resize_scale": prepared["scale"],
        "engine": "paddleocr",
        "paddle_api_used": v13_ocr.get("paddle_api_used"),
        "ocr_runtime_profile": "v13_docker_compatible",
        "ocr_lang": lang,
        "ocr_device": device,
        "text_recognition_model_name": v13_ocr.get("text_recognition_model_name"),
        "duration_seconds": round(time.perf_counter() - started, 2),
        "word_count": len(words),
        "line_count": len(lines),
        "words": words,
        "lines": lines,
    }
    save_json(out_json_path, ocr_json)
    _emit(
        progress_callback,
        "ocr_full_image",
        "done",
        "Full-image OCR finished.",
        duration_seconds=ocr_json["duration_seconds"],
        image_height=ocr_json["image_height"],
        image_width=ocr_json["image_width"],
        word_count=len(words),
        line_count=len(lines),
        paddle_api_used=ocr_json["paddle_api_used"],
    )
    return ocr_json
