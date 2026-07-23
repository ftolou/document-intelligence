#!/usr/bin/env python3
"""Apply receipt database migrations and print the resulting schema version."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from receipt_intelligence import settings  # noqa: E402
from receipt_intelligence.storage.receipt_db import ReceiptDatabase  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply receipt SQLite migrations.")
    parser.add_argument(
        "--db",
        type=Path,
        default=settings.RECEIPT_DB_PATH,
        help="SQLite database path (defaults to the application setting).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    database = ReceiptDatabase(args.db)
    print(
        json.dumps(
            {
                "db_path": str(database.db_path),
                "schema_version": database.migrations.current_version(),
                "summary": database.summary(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
