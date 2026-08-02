#!/usr/bin/env python3
"""Validate a Phase 3 structured receipt without invoking models or correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from receipt_intelligence.extraction.contracts.validation import ValidationRequest
from receipt_intelligence.extraction.validation.engine import DeterministicValidationEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path, help="Phase 3 assembled receipt JSON")
    parser.add_argument("--item-contract", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--item-pipeline-disabled", action="store_true")
    parser.add_argument("--scalar-tasks", nargs="*", default=[])
    parser.add_argument("--money-tolerance", type=float, default=0.02)
    parser.add_argument("--vat-rate-tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    item_contract = (
        json.loads(args.item_contract.read_text(encoding="utf-8"))
        if args.item_contract
        else {"status": "valid", "errors": [], "warnings": []}
    )
    report = DeterministicValidationEngine().validate(
        ValidationRequest(
            receipt=receipt,
            item_contract=item_contract,
            item_pipeline_enabled=not args.item_pipeline_disabled,
            selected_scalar_tasks=tuple(args.scalar_tasks),
            money_tolerance=args.money_tolerance,
            vat_rate_tolerance=args.vat_rate_tolerance,
        )
    ).to_dict()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
