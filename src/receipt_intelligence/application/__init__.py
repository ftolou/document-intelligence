"""Application-level contracts shared by extraction, RAG, and RAG-SQL.

The package root exposes LLM JSON helpers lazily so importing an unrelated
application submodule does not initialize LLM contracts.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from receipt_intelligence.application.llm_json import (
        LLMJsonParseError,
        parse_json_from_llm,
    )

    _TYPE_EXPORTS = (LLMJsonParseError, parse_json_from_llm)

__all__ = ["LLMJsonParseError", "parse_json_from_llm"]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.llm_json")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
