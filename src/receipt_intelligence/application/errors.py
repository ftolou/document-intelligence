"""Application-level errors that transport adapters can map to responses."""

from __future__ import annotations


class ApplicationError(Exception):
    """Base class for expected use-case failures."""

    default_code = "application_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code or self.default_code


class InvalidRequestError(ApplicationError):
    """The command is malformed or violates a use-case precondition."""

    default_code = "invalid_request"


class ServiceUnavailableError(ApplicationError):
    """A required application resource is temporarily unavailable."""

    default_code = "service_unavailable"


class ResourceNotFoundError(ApplicationError):
    """A requested application resource does not exist."""

    default_code = "not_found"


class UnsupportedResourceError(InvalidRequestError):
    """The supplied resource type is not supported by the use case."""

    default_code = "unsupported_resource"


__all__ = [
    "ApplicationError",
    "InvalidRequestError",
    "ResourceNotFoundError",
    "ServiceUnavailableError",
    "UnsupportedResourceError",
]
