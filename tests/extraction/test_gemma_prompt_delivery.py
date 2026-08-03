from __future__ import annotations

from receipt_intelligence.application.ports.chat import ChatGenerationResult
from receipt_intelligence.extraction.settings import ParsingSettings
from receipt_intelligence.extraction.structured.catalog import SCALAR_TASKS
from receipt_intelligence.extraction.structured.task_runner import GemmaTaskRunner
from receipt_intelligence.prompts.registry import default_prompt_registry


class CapturingChatGateway:
    def __init__(self) -> None:
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return ChatGenerationResult(text='{"merchant_name":"Testmarkt"}')


def test_gemma_runner_delivers_question_schema_and_receipt_evidence() -> None:
    gateway = CapturingChatGateway()
    runner = GemmaTaskRunner(
        gateway=gateway,
        prompts=default_prompt_registry(),
        settings=ParsingSettings(
            ollama_url="http://ollama",
            model="gemma4",
            scalar_tasks=("merchant_name",),
        ),
    )
    evidence = "R0001 :: Testmarkt\nR0002 :: Bonsumme 9,99"

    result = runner.run(SCALAR_TASKS["merchant_name"], evidence)

    assert result.answer == {"merchant_name": "Testmarkt"}
    assert len(gateway.requests) == 1
    request = gateway.requests[0]
    assert "What is the name of the business" in request.user_prompt
    assert '"type":"object"' in request.user_prompt
    assert evidence in request.user_prompt
    assert "$question" not in request.user_prompt
    assert "$schema_json" not in request.user_prompt
    assert "$evidence" not in request.user_prompt
