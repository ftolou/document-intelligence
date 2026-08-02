"""Canonical row serialization for Qwen receipt transcription."""

from __future__ import annotations

import re
from collections.abc import Sequence

from receipt_intelligence.extraction.contracts.transcription import (
    CanonicalTranscriptionRow,
    TranscriptionFragment,
)


def strip_code_fences(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def clean_plain_lines(text: str) -> list[str]:
    """Remove transport wrappers without validating transcription semantics."""

    lines: list[str] = []
    for raw_line in strip_code_fences(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in {"```", "```text", "```txt", "BEGIN_RECEIPT", "END_RECEIPT"}:
            continue
        line = re.sub(r"^(?:[-*]\s+)", "", line).strip()
        line = re.sub(r"^R\d{4}\s*::\s*", "", line).strip()
        if line:
            lines.append(line)
    if not lines:
        raise RuntimeError("The Qwen transcription contained no usable lines")
    return lines


def build_canonical_rows(
    fragments: Sequence[TranscriptionFragment],
) -> tuple[CanonicalTranscriptionRow, ...]:
    rows: list[CanonicalTranscriptionRow] = []
    ordered = sorted(fragments, key=lambda fragment: fragment.order)
    for fragment in ordered:
        for text in clean_plain_lines(fragment.text):
            rows.append(
                CanonicalTranscriptionRow(
                    row_id=f"R{len(rows) + 1:04d}",
                    text=text,
                    source_crop_ids=(fragment.crop_id,),
                )
            )
    if not rows:
        raise RuntimeError("Concatenated Qwen transcription is empty")
    return tuple(rows)


def serialize_canonical_rows(rows: Sequence[CanonicalTranscriptionRow]) -> str:
    output = ["BEGIN_RECEIPT"]
    output.extend(f"{row.row_id} :: {row.text}" for row in rows)
    output.append("END_RECEIPT")
    return "\n".join(output)


__all__ = [
    "build_canonical_rows",
    "clean_plain_lines",
    "serialize_canonical_rows",
    "strip_code_fences",
]
