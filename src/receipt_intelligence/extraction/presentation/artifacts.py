"""Filesystem artifact store with current UI/review-compatible filenames."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from receipt_intelligence.application.ports.artifacts import (
    ArtifactKind,
    ArtifactReference,
    ArtifactStore,
)

_FILENAMES: dict[ArtifactKind, str] = {
    ArtifactKind.TRANSCRIPTION: "{run_id}_transcription.json",
    ArtifactKind.STRUCTURED_EXTRACTION: "{run_id}_structured_extraction.json",
    ArtifactKind.INITIAL_RECEIPT: "{run_id}_89_receipt_structured_initial.json",
    ArtifactKind.INITIAL_VALIDATION: "{run_id}_89_deterministic_validation_initial.json",
    ArtifactKind.CORRECTION_REPORT: "{run_id}_90_gemma_correction_report.json",
    ArtifactKind.FINAL_VALIDATION: "{run_id}_v14_validation_report.json",
    ArtifactKind.CATEGORIZATION_PROMPT: "{run_id}_v14_14_categorization_prompt.txt",
    ArtifactKind.CATEGORIZATION_RAW: "{run_id}_v14_14_categorization_raw.txt",
    ArtifactKind.CATEGORIZATION_RESULT: "{run_id}_v14_14_categorization_result.json",
    ArtifactKind.FINAL_RECEIPT: "{run_id}_receipt_final.json",
    ArtifactKind.FINAL_RECEIPT_RECONCILED: "{run_id}_receipt_final_reconciled.json",
    ArtifactKind.FINAL_RECEIPT_CATEGORIZED: "{run_id}_receipt_final_categorized.json",
    ArtifactKind.RECONCILIATION_REPORT: "{run_id}_reconciliation_report.json",
    ArtifactKind.PIPELINE_METADATA: "{run_id}_pipeline_meta.json",
    ArtifactKind.STAGE_TRACE: "{run_id}_extraction_stage_trace.json",
}

_ALIASES: dict[ArtifactKind, str] = {
    ArtifactKind.FINAL_VALIDATION: "latest_v14_validation_report.json",
    ArtifactKind.CATEGORIZATION_PROMPT: "latest_v14_14_categorization_prompt.txt",
    ArtifactKind.CATEGORIZATION_RAW: "latest_v14_14_categorization_raw.txt",
    ArtifactKind.CATEGORIZATION_RESULT: "latest_v14_14_categorization_result.json",
    ArtifactKind.FINAL_RECEIPT: "latest_receipt_final.json",
    ArtifactKind.FINAL_RECEIPT_RECONCILED: "latest_receipt_final_reconciled.json",
    ArtifactKind.FINAL_RECEIPT_CATEGORIZED: "latest_receipt_final_categorized.json",
    ArtifactKind.RECONCILIATION_REPORT: "latest_reconciliation_report.json",
    ArtifactKind.PIPELINE_METADATA: "latest_pipeline_meta.json",
    ArtifactKind.STAGE_TRACE: "latest_extraction_stage_trace.json",
}

_PATH_KEYS: dict[ArtifactKind, str] = {
    ArtifactKind.TRANSCRIPTION: "transcription",
    ArtifactKind.STRUCTURED_EXTRACTION: "structured_extraction",
    ArtifactKind.INITIAL_RECEIPT: "receipt_structured_initial",
    ArtifactKind.INITIAL_VALIDATION: "validation_report_initial",
    ArtifactKind.CORRECTION_REPORT: "correction_report",
    ArtifactKind.FINAL_VALIDATION: "validation_report",
    ArtifactKind.CATEGORIZATION_PROMPT: "categorization_prompt",
    ArtifactKind.CATEGORIZATION_RAW: "categorization_raw",
    ArtifactKind.CATEGORIZATION_RESULT: "categorization_result",
    ArtifactKind.FINAL_RECEIPT: "receipt_final",
    ArtifactKind.FINAL_RECEIPT_RECONCILED: "receipt_final_reconciled",
    ArtifactKind.FINAL_RECEIPT_CATEGORIZED: "receipt_final_categorized",
    ArtifactKind.RECONCILIATION_REPORT: "reconciliation_report",
    ArtifactKind.PIPELINE_METADATA: "pipeline_meta",
    ArtifactKind.STAGE_TRACE: "stage_trace",
}


class CompatibilityFilesystemArtifactStore(ArtifactStore):
    def __init__(self, result_dir: Path) -> None:
        self.result_dir = Path(result_dir)

    def prepare_run(self, *, run_id: str, overwrite: bool) -> None:
        self.result_dir.mkdir(parents=True, exist_ok=True)
        if not overwrite:
            return
        for kind in _FILENAMES:
            path = self._path(run_id, kind)
            if path.exists():
                path.unlink()

    def write_json(
        self,
        *,
        run_id: str,
        kind: ArtifactKind,
        payload: Any,
    ) -> ArtifactReference:
        path = self._path(run_id, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return ArtifactReference(kind=kind, path=path, media_type="application/json")

    def write_text(
        self,
        *,
        run_id: str,
        kind: ArtifactKind,
        text: str,
    ) -> ArtifactReference:
        path = self._path(run_id, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text or "", encoding="utf-8")
        return ArtifactReference(kind=kind, path=path, media_type="text/plain")

    def publish_aliases(
        self,
        *,
        run_id: str,
        kinds: tuple[ArtifactKind, ...],
    ) -> tuple[ArtifactReference, ...]:
        references: list[ArtifactReference] = []
        for kind in kinds:
            filename = _ALIASES.get(kind)
            if filename is None:
                continue
            source = self._path(run_id, kind)
            if not source.exists():
                continue
            destination = self.result_dir / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            references.append(
                ArtifactReference(
                    kind=kind,
                    path=destination,
                    media_type=(
                        "text/plain" if destination.suffix == ".txt" else "application/json"
                    ),
                )
            )
        return tuple(references)

    @staticmethod
    def path_key(kind: ArtifactKind) -> str:
        return _PATH_KEYS[kind]

    def _path(self, run_id: str, kind: ArtifactKind) -> Path:
        run_id = str(run_id or "").strip()
        if not run_id:
            raise ValueError("run_id must not be empty.")
        try:
            template = _FILENAMES[kind]
        except KeyError as exc:
            raise ValueError(f"No filename mapping for artifact kind {kind!r}.") from exc
        return self.result_dir / template.format(run_id=run_id)


__all__ = ["CompatibilityFilesystemArtifactStore"]
