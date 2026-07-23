"""Flask application factory and HTTP routes."""

from __future__ import annotations

from typing import Any


def create_app(*args: Any, **kwargs: Any):
    """Import Flask only when the application factory is actually used."""

    from receipt_intelligence.web.app_factory import create_app as factory

    return factory(*args, **kwargs)


__all__ = ["create_app"]
