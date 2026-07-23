#!/usr/bin/env python3
"""Compatibility entry point for the Flask application factory."""

from __future__ import annotations

import receipt_intelligence.settings as settings
from receipt_intelligence.web.app_factory import create_app

app = create_app()


def main() -> None:
    app.run(
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        debug=settings.DEBUG,
        threaded=True,
    )


if __name__ == "__main__":
    main()
