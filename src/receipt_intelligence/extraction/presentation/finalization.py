"""Pure final receipt assembly plus compatibility artifact publication."""

from __future__ import annotations

import copy
from pathlib import Path

from receipt_intelligence.application.ports.artifacts import ArtifactKind, ArtifactReference
from receipt_intelligence.extraction.contracts.common import JsonObject, StageArtifact
from receipt_intelligence.extraction.contracts.presentation import (
    CategorizationResult,
    FinalizationRequest,
    FinalizationResult,
)
from receipt_intelligence.extraction.presentation.artifacts import (
    CompatibilityFilesystemArtifactStore,
)
from receipt_intelligence.extraction.services.finalization import ReceiptFinalizationService


class CompatibilityFinalizationService(ReceiptFinalizationService):
    def __init__(
        self,
        *,
        artifact_store: CompatibilityFilesystemArtifactStore,
        app_version: str,
        overwrite: bool = True,
    ) -> None:
        self._store = artifact_store
        self._app_version = str(app_version or "").strip() or "unknown"
        self._overwrite = overwrite

    def finalize(self, request: FinalizationRequest) -> FinalizationResult:
        self._store.prepare_run(run_id=request.run_id, overwrite=self._overwrite)
        final_receipt = copy.deepcopy(request.categorization.receipt or request.receipt)
        final_validation = request.validation.to_dict()
        final_receipt["validation"] = final_validation
        metadata = _pipeline_metadata(request, app_version=self._app_version)
        final_receipt["pipeline"] = {
            "architecture": metadata["architecture"],
            "app_version": self._app_version,
            "workflow": metadata["workflow"]["name"],
            "staged_execution": True,
            "read_only_validation": True,
            "no_generic_correction_fallback": True,
            "categorization_status": request.categorization.status.value,
        }

        references: list[ArtifactReference] = []
        if request.categorization.status.value != "disabled":
            references.extend(
                [
                    self._store.write_text(
                        run_id=request.run_id,
                        kind=ArtifactKind.CATEGORIZATION_PROMPT,
                        text=request.categorization.prompt,
                    ),
                    self._store.write_text(
                        run_id=request.run_id,
                        kind=ArtifactKind.CATEGORIZATION_RAW,
                        text=request.categorization.raw_output,
                    ),
                    self._store.write_json(
                        run_id=request.run_id,
                        kind=ArtifactKind.CATEGORIZATION_RESULT,
                        payload=_categorization_artifact(
                            request.categorization, app_version=self._app_version
                        ),
                    ),
                ]
            )
        references.extend(
            [
                self._store.write_json(
                    run_id=request.run_id,
                    kind=ArtifactKind.FINAL_VALIDATION,
                    payload=final_validation,
                ),
                self._store.write_json(
                    run_id=request.run_id,
                    kind=ArtifactKind.RECONCILIATION_REPORT,
                    payload=final_validation,
                ),
                self._store.write_json(
                    run_id=request.run_id,
                    kind=ArtifactKind.FINAL_RECEIPT,
                    payload=final_receipt,
                ),
                self._store.write_json(
                    run_id=request.run_id,
                    kind=ArtifactKind.FINAL_RECEIPT_RECONCILED,
                    payload=final_receipt,
                ),
                self._store.write_json(
                    run_id=request.run_id,
                    kind=ArtifactKind.FINAL_RECEIPT_CATEGORIZED,
                    payload=final_receipt,
                ),
                self._store.write_json(
                    run_id=request.run_id,
                    kind=ArtifactKind.PIPELINE_METADATA,
                    payload=metadata,
                ),
                self._store.write_json(
                    run_id=request.run_id,
                    kind=ArtifactKind.STAGE_TRACE,
                    payload=[dict(value) for value in request.stage_trace],
                ),
            ]
        )
        aliases = self._store.publish_aliases(
            run_id=request.run_id,
            kinds=tuple(reference.kind for reference in references),
        )
        all_references = tuple(references) + aliases
        paths = _paths(self._store, all_references)
        return FinalizationResult(
            receipt=final_receipt,
            validation=request.validation,
            categorization=request.categorization,
            pipeline_metadata=metadata,
            paths=paths,
            artifacts=tuple(_stage_artifact(reference) for reference in all_references),
        )


def _pipeline_metadata(request: FinalizationRequest, *, app_version: str) -> JsonObject:
    upstream = copy.deepcopy(request.upstream_metadata)
    stage_trace = [dict(value) for value in request.stage_trace]
    failed_codes = sorted(request.validation.failed_codes)
    categorization = request.categorization
    receipt_categorization = (
        categorization.receipt.get("categorization")
        if isinstance(categorization.receipt.get("categorization"), dict)
        else {}
    )
    metadata: JsonObject = {
        "schema_version": "next_pipeline_meta_1",
        "app_version": app_version,
        "architecture": (
            "Paddle geometry -> Qwen canonical transcription -> Gemma scalar/item extraction "
            "-> read-only deterministic validation -> source-evidence specialist correction "
            "-> optional item categorization -> final publication"
        ),
        "workflow": {
            "name": "NextReceiptExtractionWorkflow",
            "staged_execution": True,
            "stage_count": len(stage_trace),
            "stages": [str(value.get("stage") or "") for value in stage_trace],
            "stage_trace": stage_trace,
        },
        "safety": {
            "read_only_validation": True,
            "no_deterministic_semantic_parser": True,
            "no_generic_correction_fallback": True,
            "correction_requires_target_resolution": True,
            "correction_rejects_regressions": True,
        },
        "validation": {
            "status": request.validation.status,
            "failed_codes": failed_codes,
            "failure_count": len(failed_codes),
        },
        "categorization": {
            "status": categorization.status.value,
            "model": categorization.model,
            "duration_seconds": categorization.duration_seconds,
            "warning_count": len(categorization.warnings),
            "error": categorization.error,
            "item_count": receipt_categorization.get("item_count"),
            "categorized_count": receipt_categorization.get("categorized_count"),
            "category_review_count": receipt_categorization.get("category_review_count"),
            "merchant_classification": dict(categorization.merchant_classification),
        },
    }
    for key in ("transcription", "extraction", "correction"):
        value = upstream.pop(key, None)
        if isinstance(value, dict):
            metadata[key] = value
    if upstream:
        metadata["additional_context"] = upstream
    return metadata


def _categorization_artifact(
    categorization: CategorizationResult, *, app_version: str
) -> JsonObject:
    receipt_meta = (
        categorization.receipt.get("categorization")
        if isinstance(categorization.receipt.get("categorization"), dict)
        else {}
    )
    return {
        "schema_version": receipt_meta.get("schema_version") or "v14_14_item_categories_1",
        "app_version": app_version,
        "status": categorization.status.value,
        "categories": [dict(value) for value in categorization.categories],
        "merchant_classification": dict(categorization.merchant_classification),
        "warnings": list(categorization.warnings),
        "duration_seconds": categorization.duration_seconds,
        "error": categorization.error,
    }


def _paths(
    store: CompatibilityFilesystemArtifactStore,
    references: tuple[ArtifactReference, ...],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for reference in references:
        key = store.path_key(reference.kind)
        if reference.path.name.startswith("latest_"):
            key = f"latest_{key}"
        paths[key] = reference.path
    return paths


def _stage_artifact(reference: ArtifactReference) -> StageArtifact:
    return StageArtifact(
        name=reference.kind.value,
        path=reference.path,
        media_type=reference.media_type,
    )


__all__ = ["CompatibilityFinalizationService"]
