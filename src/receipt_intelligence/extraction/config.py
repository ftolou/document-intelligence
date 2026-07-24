"""Typed configuration contracts for receipt extraction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Immutable inputs for one receipt extraction run.

    This contract is intentionally explicit. Runtime callers must provide only
    declared fields; compatibility aliases are handled outside the workflow by
    the legacy entry-point adapter.
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

    gpu_orchestration: str = "none"
    unload_llm_before_vlm: bool = False
    reload_llm_after_vlm: bool = False
    ollama_control_mode: str = "api"
    ollama_control_timeout_seconds: float = 120.0
    ollama_unload_command: str = ""
    ollama_start_command: str = ""
    ollama_reload_prompt: str = "ok"
    ollama_gpu_handoff_wait_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "ocr_json_path", Path(self.ocr_json_path))
        object.__setattr__(self, "result_dir", Path(self.result_dir))
        if self.source_image_path is not None:
            object.__setattr__(self, "source_image_path", Path(self.source_image_path))
        object.__setattr__(
            self,
            "gpu_orchestration",
            (self.gpu_orchestration or "none").strip().lower(),
        )
        object.__setattr__(
            self,
            "ollama_control_mode",
            (self.ollama_control_mode or "api").strip().lower(),
        )

        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.ollama_url.strip():
            raise ValueError("ollama_url must not be empty")


@dataclass(frozen=True, slots=True)
class ExtractionRequest(ExtractionConfig):
    """Public typed request consumed by the extraction application service."""
