"""Schema-constrained Gemma task invocation."""

from __future__ import annotations

import json

from receipt_intelligence.application.llm_json import parse_json_from_llm
from receipt_intelligence.application.ports.chat import ChatGateway, ChatGenerationRequest
from receipt_intelligence.extraction.contracts.extraction import (
    GemmaTaskResult,
    GemmaTaskStatus,
)
from receipt_intelligence.extraction.settings import ParsingSettings
from receipt_intelligence.extraction.structured.catalog import (
    SYSTEM_PROMPT,
    TASK_ENVELOPE,
    TaskDefinition,
)
from receipt_intelligence.extraction.structured.normalization import normalize_task_answer
from receipt_intelligence.prompts.registry import PromptRegistry


class GemmaTaskRunner:
    def __init__(
        self,
        *,
        gateway: ChatGateway,
        prompts: PromptRegistry,
        settings: ParsingSettings,
    ) -> None:
        self._gateway = gateway
        self._prompts = prompts
        self._settings = settings

    def run(self, definition: TaskDefinition, evidence: str) -> GemmaTaskResult:
        question = self._prompts.read_template(definition.prompt)
        schema = self._prompts.load_schema(definition.prompt)
        if schema is None:
            raise RuntimeError(
                f"Gemma task has no registered schema: {definition.prompt.prompt_id}"
            )
        user_prompt = self._prompts.render(
            TASK_ENVELOPE,
            question=question,
            schema_json=json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
            evidence=evidence,
        )
        result = self._gateway.generate(
            ChatGenerationRequest(
                model=self._settings.model,
                system_prompt=self._prompts.read_template(SYSTEM_PROMPT),
                user_prompt=user_prompt,
                response_json_schema=schema,
                operation=f"receipt_{definition.task_name}",
                think=self._settings.think,
                num_ctx=self._settings.num_ctx,
                num_predict=(
                    self._settings.item_num_predict
                    if definition.task_name == "direct_receipt_items"
                    else definition.num_predict
                ),
                temperature=self._settings.temperature,
                seed=self._settings.seed,
                keep_alive=self._settings.keep_alive,
                timeout_seconds=self._settings.timeout_seconds,
            )
        )
        answer = parse_json_from_llm(result, response_json_schema=schema)
        answer, normalization_changes = normalize_task_answer(
            task_name=definition.task_name,
            answer=answer,
            schema=schema,
            evidence=evidence,
        )
        return GemmaTaskResult(
            task_name=definition.task_name,
            prompt_id=definition.prompt.prompt_id,
            status=GemmaTaskStatus.COMPLETED,
            answer=answer,
            raw_model_content=result.text,
            thinking=result.thinking,
            metrics=result.metrics,
            diagnostics={
                "prompt_version": definition.prompt.version,
                "schema_delivery": "provider_neutral_json_schema",
                "think": self._settings.think,
                "num_predict": (
                    self._settings.item_num_predict
                    if definition.task_name == "direct_receipt_items"
                    else definition.num_predict
                ),
                "normalization_changes": list(normalization_changes),
            },
        )


__all__ = ["GemmaTaskRunner"]
