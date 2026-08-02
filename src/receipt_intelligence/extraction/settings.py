"""Grouped immutable settings for the next receipt extraction pipeline.

The active workflow still consumes :class:`ExtractionConfig`. These grouped settings are an
additive migration target. Later phases can move one stage at a time and use
``PipelineSettings.from_extraction_config`` at the compatibility boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from receipt_intelligence.extraction.config import ExtractionConfig


@dataclass(frozen=True, slots=True)
class DetectionSettings:
    language: str = "german"
    device: str = "cpu"
    max_crops: int = 8
    max_side_length: int | None = None


@dataclass(frozen=True, slots=True)
class TranscriptionSettings:
    ollama_url: str
    model: str
    num_ctx: int = 8192
    num_predict: int = 4096
    timeout_seconds: float = 300.0
    keep_alive: str | None = None
    think: bool = False


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
                max_crops=config.max_reocr_images or 8,
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
        )


__all__ = [
    "CategorizationSettings",
    "CorrectionSettings",
    "DetectionSettings",
    "ParsingSettings",
    "PipelineSettings",
    "RuntimeSettings",
    "TranscriptionSettings",
    "ValidationSettings",
]
