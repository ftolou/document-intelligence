"""Gemma scalar/item extraction service with pure receipt assembly."""

from __future__ import annotations

import concurrent.futures
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from receipt_intelligence.extraction.contracts.common import StageArtifact
from receipt_intelligence.extraction.contracts.extraction import (
    GemmaTaskResult,
    GemmaTaskStatus,
    StructuredExtractionRequest,
    StructuredExtractionResult,
)
from receipt_intelligence.extraction.services.structured_extraction import (
    StructuredExtractionService,
)
from receipt_intelligence.extraction.settings import ParsingSettings
from receipt_intelligence.extraction.structured.assembler import assemble_receipt
from receipt_intelligence.extraction.structured.catalog import ITEM_TASK, SCALAR_TASKS
from receipt_intelligence.extraction.structured.item_contract import validate_direct_items
from receipt_intelligence.extraction.structured.task_runner import GemmaTaskRunner


class GemmaStructuredExtractionService(StructuredExtractionService):
    def __init__(
        self,
        *,
        task_runner: GemmaTaskRunner,
        settings: ParsingSettings,
        result_dir: Path,
    ) -> None:
        self._runner = task_runner
        self._settings = settings
        self._result_dir = Path(result_dir)

    def extract(self, request: StructuredExtractionRequest) -> StructuredExtractionResult:
        started = time.perf_counter()
        work_dir = self._result_dir / f"{request.run_id}_next_structured_extraction"
        work_dir.mkdir(parents=True, exist_ok=True)
        evidence = request.transcription.canonical_text
        selected = () if self._settings.skip_scalars else self._settings.scalar_tasks
        unknown = [name for name in selected if name not in SCALAR_TASKS]
        if unknown:
            raise ValueError("Unknown scalar tasks: " + ", ".join(unknown))

        scalar_results = self._run_scalars(selected, evidence, work_dir)
        item_result = None if self._settings.skip_items else self._run_item(evidence, work_dir)
        item_contract = validate_direct_items(item_result.answer if item_result else None)
        if self._settings.skip_items:
            item_contract = {
                "status": "skipped",
                "errors": [],
                "warnings": [],
                "observations": [],
                "metrics": {},
            }
        receipt = assemble_receipt(scalar_results, item_result)
        missing = tuple(
            name
            for name in selected
            if not any(
                result.task_name == name and result.status is GemmaTaskStatus.COMPLETED
                for result in scalar_results
            )
        )
        receipt_path = work_dir / "70_receipt_structured_initial.json"
        report_path = work_dir / "70_structured_extraction_report.json"
        self._write_json(receipt_path, receipt)
        diagnostics = {
            "status": "completed"
            if not missing and item_contract.get("status") != "invalid"
            else "completed_with_errors",
            "strategy": "gemma_scalar_specialists_plus_direct_items",
            "selected_scalar_tasks": list(selected),
            "missing_or_failed_scalar_tasks": list(missing),
            "item_pipeline_enabled": not self._settings.skip_items,
            "item_contract_status": item_contract.get("status"),
            "deterministic_validation_used": False,
            "semantic_correction_used": False,
            "duration_seconds": round(time.perf_counter() - started, 3),
        }
        self._write_json(
            report_path,
            {
                **diagnostics,
                "scalar_results": [_task_payload(value) for value in scalar_results],
                "item_result": _task_payload(item_result) if item_result else None,
                "item_contract": item_contract,
            },
        )
        return StructuredExtractionResult(
            receipt=receipt,
            scalar_results=scalar_results,
            item_result=item_result,
            item_contract=item_contract,
            missing_scalar_tasks=missing,
            diagnostics=diagnostics,
            artifacts=(
                StageArtifact(name="next_structured_receipt", path=receipt_path),
                StageArtifact(name="next_structured_extraction_report", path=report_path),
            ),
        )

    def _run_scalars(
        self,
        selected: tuple[str, ...],
        evidence: str,
        work_dir: Path,
    ) -> tuple[GemmaTaskResult, ...]:
        if not selected:
            return ()
        results: dict[str, GemmaTaskResult] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(self._settings.parallelism, len(selected)))
        ) as executor:
            futures = {
                executor.submit(self._safe_run, SCALAR_TASKS[name], evidence): name
                for name in selected
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                result = future.result()
                results[name] = result
                self._write_json(work_dir / f"scalar_{name}.json", _task_payload(result))
        return tuple(results[name] for name in selected)

    def _run_item(self, evidence: str, work_dir: Path) -> GemmaTaskResult:
        result = self._safe_run(ITEM_TASK, evidence)
        self._write_json(work_dir / "60_gemma_direct_items.json", _task_payload(result))
        if result.thinking:
            (work_dir / "60_gemma_direct_items_thinking.txt").write_text(
                result.thinking + "\n", encoding="utf-8"
            )
        return result

    def _safe_run(self, definition: Any, evidence: str) -> GemmaTaskResult:
        try:
            return self._runner.run(definition, evidence)
        except Exception as exc:
            return GemmaTaskResult(
                task_name=definition.task_name,
                prompt_id=definition.prompt.prompt_id,
                status=GemmaTaskStatus.ERROR,
                error=f"{type(exc).__name__}: {exc}",
                diagnostics={"prompt_version": definition.prompt.version},
            )

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )


def _task_payload(result: GemmaTaskResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "task": result.task_name,
        "prompt_id": result.prompt_id,
        "status": result.status.value,
        "answer": result.answer,
        "raw_model_content": result.raw_model_content,
        "thinking": result.thinking,
        "metrics": result.metrics.to_diagnostics() if result.metrics else None,
        "error": result.error,
        "diagnostics": result.diagnostics,
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


__all__ = ["GemmaStructuredExtractionService"]
