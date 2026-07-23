#!/usr/bin/env python3
"""Build a table-interpretation prompt from a saved visual evidence JSON.

This script does not call Ollama. It is a lightweight smoke/demo helper for the
Phase 2.1 table-interpretation stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from receipt_intelligence.extraction.parsing.table_interpreter import (
    build_table_interpretation_prompt,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a table interpretation prompt from saved visual evidence"
    )
    parser.add_argument(
        "visual_evidence",
        type=Path,
        help="Path to latest_v14_6_visual_evidence.json or run-specific visual evidence JSON",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional output prompt path")
    args = parser.parse_args()

    evidence = json.loads(args.visual_evidence.read_text(encoding="utf-8-sig"))
    if not isinstance(evidence, dict):
        raise SystemExit("visual evidence root must be a JSON object")
    prompt = build_table_interpretation_prompt(evidence)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(prompt, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
