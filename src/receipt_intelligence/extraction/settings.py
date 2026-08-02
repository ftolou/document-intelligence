"""Grouped immutable settings for the next receipt extraction pipeline.

The active workflow still consumes :class:`ExtractionConfig`. These grouped settings are an
additive migration target. Later phases can move one stage at a time and use
``PipelineSettings.from_extraction_config`` at the compatibility boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from receipt_intelligence.extraction.config import ExtractionConfig


@dataclass(frozen=True, slots=True)
class DetectionSettings:
    language: str = "en"
    device: str = "cpu"
    max_crops: int = 8
    max_side_length: int | None = None
    backend: str = "auto"
    model_name: str | None = None
    minimum_score: float = 0.20
    minimum_box_width: float = 6.0
    minimum_box_height: float = 4.0
    minimum_line_width: float = 8.0
    minimum_line_height: float = 4.0
    line_overlap_threshold: float = 0.40
    line_center_factor: float = 0.60
    minimum_lines: int = 3
    maximum_lines: int = 300

    def __post_init__(self) -> None:
        backend = str(self.backend or "").strip().lower()
        if backend not in {"auto", "text_detection", "paddleocr"}:
            raise ValueError("DetectionSettings.backend is not supported.")
        if self.max_side_length is not None and self.max_side_length < 1:
            raise ValueError("DetectionSettings.max_side_length must be positive.")
        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError("DetectionSettings.minimum_score must be between 0 and 1.")
        if self.minimum_lines < 0 or self.maximum_lines < self.minimum_lines:
            raise ValueError("DetectionSettings line-count bounds are invalid.")
        object.__setattr__(self, "backend", backend)


@dataclass(frozen=True, slots=True)
class CropPlanningSettings:
    max_crops: int = 4
    target_rows_per_crop: int = 18
    single_crop_max_rows: int = 25
    single_crop_max_aspect_ratio: float = 2.0
    minimum_lines_per_crop: int = 3
    safe_cut_search_ratio: float = 0.20
    maximum_safe_cut_search_ratio: float = 0.35
    full_width_crops: bool = True
    horizontal_padding: int = 20
    minimum_safe_gap: float = 1.0
    cut_search_margin: int = 1
    cut_strip_half_height: int = 3
    cut_ink_threshold: int = 190
    maximum_cut_ink_density: float = 0.01

    def __post_init__(self) -> None:
        if self.max_crops < 1:
            raise ValueError("CropPlanningSettings.max_crops must be positive.")
        if self.target_rows_per_crop < 1 or self.single_crop_max_rows < 1:
            raise ValueError("CropPlanningSettings row limits must be positive.")
        if self.minimum_lines_per_crop < 1:
            raise ValueError("minimum_lines_per_crop must be positive.")
        if not (
            0.0
            <= self.safe_cut_search_ratio
            <= self.maximum_safe_cut_search_ratio
            <= 1.0
        ):
            raise ValueError("Safe-cut search ratios are invalid.")
        if not 0 <= self.cut_ink_threshold <= 255:
            raise ValueError("cut_ink_threshold must be between 0 and 255.")
        if not 0.0 <= self.maximum_cut_ink_density <= 1.0:
            raise ValueError("maximum_cut_ink_density must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class TranscriptionSettings:
    ollama_url: str
    model: str
    num_ctx: int = 8192
    num_predict: int = 4096
    timeout_seconds: float = 300.0
    keep_alive: str | None = None
    think: bool = False
    temperature: float = 0.0
    seed: int = 42
    parallelism: int = 1
    retries: int = 1
    crop_scale: float = 1.0
    crop_contrast: float = 1.15
    crop_sharpen: bool = False

    def __post_init__(self) -> None:
        if not str(self.ollama_url or "").strip():
            raise ValueError("TranscriptionSettings.ollama_url must not be empty.")
        if not str(self.model or "").strip():
            raise ValueError("TranscriptionSettings.model must not be empty.")
        if self.parallelism < 1 or self.retries < 0:
            raise ValueError("Transcription parallelism/retries are invalid.")
        if self.crop_scale <= 0 or self.crop_contrast <= 0:
            raise ValueError("Transcription crop preprocessing values must be positive.")


@dataclass(frozen=True, slots=True)
class ParsingSettings:
    ollama_url: str
    model: str
    num_ctx: int = 24384
    num_predict: int = 8192
    timeout_seconds: float = 300.0
    json_retry_count: int = 1
    format_json: bool = True


@dataclass(frozen=True, slots=True)
class ValidationSettings:
    tolerance: float = 0.03


@dataclass(frozen=True, slots=True)
class CorrectionSettings:
    """Correction activation plus an optional external strategy profile.

    Strategy routes, prompt references, retries, token limits, and thinking settings remain in
    the versioned correction profile rather than being duplicated in Python configuration.
    """

    enabled: bool = True
    profile_path: Path | None = None

    def __post_init__(self) -> None:
        if self.profile_path is not None:
            object.__setattr__(self, "profile_path", Path(self.profile_path))


@dataclass(frozen=True, slots=True)
class CategorizationSettings:
    enabled: bool = True
    model: str | None = None
    num_ctx: int = 8192
    num_predict: int = 4096
    timeout_seconds: float = 180.0
    format_json: bool = True


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    result_dir: Path
    run_id: str
    keep_alive: str | None = None
    gpu_orchestration: str = "none"
    unload_llm_before_multimodal: bool = False
    reload_llm_after_multimodal: bool = False

    def __post_init__(self) -> None:
        run_id = str(self.run_id or "").strip()
        if not run_id:
            raise ValueError("RuntimeSettings.run_id must not be empty.")
        object.__setattr__(self, "result_dir", Path(self.result_dir))
        object.__setattr__(self, "run_id", run_id)


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    source_image_path: Path
    legacy_ocr_json_path: Path | None
    detection: DetectionSettings
    transcription: TranscriptionSettings
    parsing: ParsingSettings
    validation: ValidationSettings
    correction: CorrectionSettings
    categorization: CategorizationSettings
    runtime: RuntimeSettings
    crop_planning: CropPlanningSettings = field(default_factory=CropPlanningSettings)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_image_path", Path(self.source_image_path))
        if self.legacy_ocr_json_path is not None:
            object.__setattr__(self, "legacy_ocr_json_path", Path(self.legacy_ocr_json_path))

    @classmethod
    def from_extraction_config(
        cls,
        config: ExtractionConfig,
        *,
        transcription_model: str,
    ) -> PipelineSettings:
        """Build grouped settings without changing the current public request contract.

        The transcription model is explicit because the legacy request has only one model field,
        while the next pipeline deliberately separates Qwen transcription from Gemma parsing.
        """

        transcription_model = str(transcription_model or "").strip()
        if not transcription_model:
            raise ValueError("transcription_model must not be empty.")
        if config.source_image_path is None:
            raise ValueError("The next pipeline requires source_image_path.")

        return cls(
            source_image_path=config.source_image_path,
            legacy_ocr_json_path=config.ocr_json_path,
            detection=DetectionSettings(
                language=config.ocr_lang,
                device=config.ocr_device,
                max_side_length=None,
            ),
            transcription=TranscriptionSettings(
                ollama_url=config.ollama_url,
                model=transcription_model,
                keep_alive=config.keep_alive,
            ),
            parsing=ParsingSettings(
                ollama_url=config.ollama_url,
                model=config.model,
                num_ctx=config.num_ctx,
                num_predict=config.num_predict,
                timeout_seconds=config.llm_timeout_seconds,
                json_retry_count=config.json_retry_count,
                format_json=config.format_json,
            ),
            validation=ValidationSettings(tolerance=config.tolerance),
            correction=CorrectionSettings(enabled=config.correction_enabled),
            categorization=CategorizationSettings(
                enabled=config.categorization_enabled,
                model=config.categorization_model,
                num_ctx=config.categorization_num_ctx,
                num_predict=config.categorization_num_predict,
                timeout_seconds=config.categorization_timeout_seconds,
                format_json=config.categorization_format_json,
            ),
            runtime=RuntimeSettings(
                result_dir=config.result_dir,
                run_id=config.run_id,
                keep_alive=config.keep_alive,
                gpu_orchestration=config.gpu_orchestration,
                unload_llm_before_multimodal=config.unload_llm_before_vlm,
                reload_llm_after_multimodal=config.reload_llm_after_vlm,
            ),
            crop_planning=CropPlanningSettings(
                max_crops=config.max_reocr_images or 4,
            ),
        )


__all__ = [
    "CategorizationSettings",
    "CorrectionSettings",
    "CropPlanningSettings",
    "DetectionSettings",
    "ParsingSettings",
    "PipelineSettings",
    "RuntimeSettings",
    "TranscriptionSettings",
    "ValidationSettings",
]
