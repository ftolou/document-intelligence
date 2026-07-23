"""Configuration model for the staged receipt extraction workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class ExtractionConfig:
    """Immutable inputs for one receipt extraction run.

    The field names intentionally match the historical
    ``run_integrated_receipt_pipeline`` keyword arguments so the public API can
    remain backward compatible while the implementation is split into stages.
    """

    ocr_json_path: Path
    result_dir: Path
    run_id: str
    ollama_url: str
    model: str

    tolerance: float = 0.03
    skip_row_llm: bool = False
    active_line_repair: bool = False
    max_repair_passes: int = 0
    max_repair_rois: int = 0
    max_repair_variants: int = 0
    max_reocr_images: int = 0
    repair_time_budget_seconds: float = 0.0
    repair_ocr_min_score: float = 0.20
    ocr_lang: str = "german"
    ocr_device: str = "cpu"
    ocr_det_model: str | None = None
    ocr_rec_model: str | None = "latin_PP-OCRv5_mobile_rec"
    progress_callback: ProgressCallback | None = None

    max_lines_for_llm: int = 260
    num_ctx: int = 24384
    num_predict: int = 8192
    keep_alive: str | None = None
    llm_timeout_seconds: float = 240.0
    json_retry_count: int = 1
    format_json: bool = True

    source_image_path: Path | None = None
    vlm_enabled: bool = False
    vlm_backend: str = "http_service"
    vlm_service_url: str = "http://receipt-vlm:7870"
    vlm_command: str = ""
    vlm_timeout_seconds: float = 180.0
    vlm_max_chars: int = 12000

    correction_enabled: bool = True
    categorization_enabled: bool = True
    categorization_model: str | None = None
    categorization_num_ctx: int = 8192
    categorization_num_predict: int = 4096
    categorization_timeout_seconds: float = 180.0
    categorization_format_json: bool = True

    unused_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def gpu_orchestration_mode(self) -> str:
        # The current VLM-first architecture intentionally avoids an automatic
        # unload/reload cycle. These accessors preserve historical knobs without
        # scattering dictionary lookups through stages.
        return str(self.unused_kwargs.get("gpu_orchestration_mode") or "none")

    @property
    def unload_before_vlm(self) -> bool:
        return bool(self.unused_kwargs.get("unload_before_vlm", False))

    @property
    def reload_after_vlm(self) -> bool:
        return bool(self.unused_kwargs.get("reload_after_vlm", False))

    @property
    def ollama_control_mode(self) -> str:
        return str(self.unused_kwargs.get("ollama_control_mode") or "api")

    @property
    def ollama_control_timeout_seconds(self) -> float:
        return float(self.unused_kwargs.get("ollama_control_timeout_seconds") or 120.0)

    @property
    def ollama_unload_command(self) -> str:
        return str(self.unused_kwargs.get("ollama_unload_command") or "")

    @property
    def ollama_start_command(self) -> str:
        return str(self.unused_kwargs.get("ollama_start_command") or "")

    @property
    def ollama_reload_prompt(self) -> str:
        return str(self.unused_kwargs.get("ollama_reload_prompt") or "ok")

    @property
    def ollama_gpu_handoff_wait_seconds(self) -> float:
        return float(self.unused_kwargs.get("ollama_gpu_handoff_wait_seconds") or 0.0)
