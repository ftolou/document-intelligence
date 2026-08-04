"""Prompt-bound source-evidence specialist calls and format-only JSON repair."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from receipt_intelligence.application.ports.chat import ChatGateway, ChatGenerationRequest
from receipt_intelligence.application.ports.llm import ModelCallMetrics
from receipt_intelligence.extraction.correction.profile import StrategyConfig
from receipt_intelligence.extraction.settings import CorrectionSettings
from receipt_intelligence.prompts.registry import PromptReference, PromptRegistry

SYSTEM_PROMPT = PromptReference("gemma.system.receipt_interpreter", "1.0.0")
JSON_REPAIR_PROMPT = PromptReference("gemma.correction.json_repair", "1.0.0")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class SourceEvidenceInvoker(Protocol):
    def invoke(
        self,
        strategy: StrategyConfig,
        transcription: str,
        round_index: int,
        attempt: int,
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class PromptBoundSourceEvidenceInvoker(SourceEvidenceInvoker):
    gateway: ChatGateway
    prompts: PromptRegistry
    settings: CorrectionSettings

    def invoke(
        self,
        strategy: StrategyConfig,
        transcription: str,
        round_index: int,
        attempt: int,
    ) -> dict[str, Any]:
        del round_index
        reference = PromptReference(strategy.prompt_id, strategy.prompt_version)
        schema = self.prompts.load_schema(reference)
        if schema is None:
            raise RuntimeError(f"Correction strategy has no schema: {strategy.strategy_id}")
        prompt = self.prompts.render(
            reference,
            source_evidence=transcription,
            output_schema=json.dumps(schema, ensure_ascii=False, indent=2),
        )
        num_predict = self._num_predict(strategy.strategy_id)
        think = self._think(strategy.strategy_id)
        generated = self.gateway.generate(
            ChatGenerationRequest(
                model=self.settings.model,
                system_prompt=self.prompts.read_template(SYSTEM_PROMPT),
                user_prompt=prompt,
                response_json_schema=None,
                operation=f"receipt_correction_{strategy.strategy_id}",
                attempt=attempt,
                think=think,
                num_ctx=self.settings.num_ctx,
                num_predict=num_predict,
                temperature=self.settings.temperature,
                seed=self.settings.seed,
                keep_alive=self.settings.keep_alive,
                timeout_seconds=self.settings.timeout_seconds,
            )
        )
        content = _strip_code_fences(generated.text)
        repair: dict[str, Any] = {"triggered": False, "status": "not_needed"}
        repaired_content: str | None = None
        try:
            answer = json.loads(content)
        except json.JSONDecodeError as original_error:
            done_reason = _done_reason(generated)
            if done_reason != "stop":
                return self._result(
                    strategy,
                    generated.text,
                    generated.metrics,
                    status="invalid_json",
                    answer=None,
                    repaired_content=None,
                    repair={
                        "triggered": False,
                        "status": "skipped_non_stop_done_reason",
                        "done_reason": done_reason,
                    },
                    error={
                        "code": "INVALID_JSON_REPAIR_SKIPPED",
                        "message": f"Invalid JSON repair skipped because done_reason={done_reason!r}.",
                        "json_error": str(original_error),
                    },
                )
            try:
                repaired = self.gateway.generate(
                    ChatGenerationRequest(
                        model=self.settings.model,
                        system_prompt=None,
                        user_prompt=self.prompts.render(
                            JSON_REPAIR_PROMPT,
                            invalid_json=content,
                        ),
                        response_json_schema=schema,
                        operation=f"receipt_correction_json_repair_{strategy.strategy_id}",
                        attempt=1,
                        think=False,
                        num_ctx=self.settings.num_ctx,
                        num_predict=num_predict,
                        temperature=0.0,
                        seed=self.settings.seed,
                        keep_alive=self.settings.keep_alive,
                        timeout_seconds=self.settings.timeout_seconds,
                    )
                )
                repaired_content = repaired.text
                answer = json.loads(_strip_code_fences(repaired.text))
                repair = {
                    "triggered": True,
                    "status": "completed",
                    "original_json_error": str(original_error),
                    "metrics": _metrics(repaired.metrics),
                }
            except Exception as repair_error:
                return self._result(
                    strategy,
                    generated.text,
                    generated.metrics,
                    status="invalid_json",
                    answer=None,
                    repaired_content=repaired_content,
                    repair={
                        "triggered": True,
                        "status": "failed",
                        "error_type": type(repair_error).__name__,
                        "error": str(repair_error),
                    },
                    error={
                        "code": "INVALID_JSON_REPAIR_FAILED",
                        "message": str(repair_error),
                        "json_error": str(original_error),
                    },
                )
        if not isinstance(answer, dict):
            return self._result(
                strategy,
                generated.text,
                generated.metrics,
                status="schema_invalid",
                answer=None,
                repaired_content=repaired_content,
                repair=repair,
                error={"code": "SOURCE_EVIDENCE_ROOT_NOT_OBJECT"},
            )
        return self._result(
            strategy,
            generated.text,
            generated.metrics,
            status="completed",
            answer=answer,
            repaired_content=repaired_content,
            repair=repair,
            error=None,
        )

    def _num_predict(self, strategy_id: str) -> int:
        if strategy_id == "item_sum_source_blocks_v3":
            return self.settings.item_sum_num_predict
        if strategy_id == "vat_source_evidence_v9":
            return self.settings.vat_num_predict
        if strategy_id == "final_total_source_evidence_v2_4":
            return self.settings.final_total_num_predict
        raise ValueError(f"Unknown correction strategy: {strategy_id}")

    def _think(self, strategy_id: str) -> bool:
        if strategy_id == "item_sum_source_blocks_v3":
            return self.settings.item_sum_think
        if strategy_id == "vat_source_evidence_v9":
            return self.settings.vat_think
        if strategy_id == "final_total_source_evidence_v2_4":
            return self.settings.final_total_think
        raise ValueError(f"Unknown correction strategy: {strategy_id}")

    def _result(
        self,
        strategy: StrategyConfig,
        raw_content: str,
        metrics: ModelCallMetrics | None,
        *,
        status: str,
        answer: dict[str, Any] | None,
        repaired_content: str | None,
        repair: dict[str, Any],
        error: dict[str, Any] | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": status,
            "task": strategy.strategy_id,
            "model": self.settings.model,
            "answer": answer,
            "raw_model_content": raw_content,
            "repaired_model_content": repaired_content,
            "json_repair": repair,
            "request": {
                "prompt": {
                    "id": strategy.prompt_id,
                    "version": strategy.prompt_version,
                },
                "schema_delivery": "embedded_once_in_prompt",
                "ollama_format_field": False,
                "current_structured_result_supplied_to_model": False,
                "think": self._think(strategy.strategy_id),
                "temperature": self.settings.temperature,
                "seed": self.settings.seed,
                "num_ctx": self.settings.num_ctx,
                "num_predict": self._num_predict(strategy.strategy_id),
            },
            "metrics": _metrics(metrics),
        }
        if error is not None:
            result["error"] = error
        return result


def _strip_code_fences(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = _FENCE.sub("", text).strip()
    return text


def _done_reason(result: Any) -> str | None:
    if result.metrics is not None and result.metrics.done_reason:
        return result.metrics.done_reason
    raw = result.raw_response if isinstance(result.raw_response, dict) else {}
    return str(raw.get("done_reason") or "").strip() or None


def _metrics(value: ModelCallMetrics | None) -> dict[str, Any] | None:
    return value.to_diagnostics() if value is not None else None


__all__ = [
    "PromptBoundSourceEvidenceInvoker",
    "SourceEvidenceInvoker",
]
