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
    declared fields.
    """

    source_image_path: Path
    result_dir: Path
    run_id: str
    ollama_url: str
    model: str

    extraction_backend: str = "local_specialized"
    source_image_max_width: int = 12000
    source_image_max_height: int = 12000
    source_image_max_pixels: int = 40_000_000
    openai_model: str = "gpt-5.6-luna"
    openai_reasoning_effort: str = "medium"
    openai_image_detail: str = "high"
    openai_max_output_tokens: int = 12000
    openai_timeout_seconds: float = 180.0

    tolerance: float = 0.03
    max_crops: int = 4
    ocr_lang: str = "german"
    ocr_device: str = "cpu"
    progress_callback: ProgressCallback | None = None
    num_ctx: int = 16384
    num_predict: int = 8192
    keep_alive: str | None = None
    llm_timeout_seconds: float = 300.0
    json_retry_count: int = 1
    format_json: bool = True

    correction_enabled: bool = True
    categorization_enabled: bool = True
    categorization_model: str | None = None
    categorization_num_ctx: int = 16384
    categorization_num_predict: int = 4096
    categorization_timeout_seconds: float = 180.0
    categorization_format_json: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_image_path", Path(self.source_image_path))
        object.__setattr__(self, "result_dir", Path(self.result_dir))
        backend = str(self.extraction_backend or "").strip().lower()
        object.__setattr__(self, "extraction_backend", backend)
        # Gemma-backed stages deliberately share one context size. Ollama treats
        # differing context sizes as different runners, even for the same model.
        object.__setattr__(self, "categorization_num_ctx", self.num_ctx)
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if backend not in {"local_specialized", "openai_one_shot"}:
            raise ValueError(f"Unsupported extraction_backend: {backend!r}")
        if backend == "local_specialized":
            if not self.model.strip():
                raise ValueError("model must not be empty")
            if not self.ollama_url.strip():
                raise ValueError("ollama_url must not be empty")
        if backend == "openai_one_shot" and not self.openai_model.strip():
            raise ValueError("openai_model must not be empty")
        for field_name in (
            "source_image_max_width",
            "source_image_max_height",
            "source_image_max_pixels",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be positive")
        if self.openai_reasoning_effort not in {"none", "minimal", "low", "medium", "high"}:
            raise ValueError("openai_reasoning_effort must be none/minimal/low/medium/high")
        if self.openai_image_detail not in {"low", "high", "auto"}:
            raise ValueError("openai_image_detail must be low/high/auto")
        if self.openai_max_output_tokens < 1:
            raise ValueError("openai_max_output_tokens must be positive")
        if self.openai_timeout_seconds <= 0:
            raise ValueError("openai_timeout_seconds must be positive")
        if self.max_crops < 1:
            raise ValueError("max_crops must be positive")


@dataclass(frozen=True, slots=True)
class ExtractionRequest(ExtractionConfig):
    """Public typed request consumed by the extraction application service."""
