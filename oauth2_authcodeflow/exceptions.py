"""Exceptions raised by the library.

``OIDCSessionEnded`` and ``OIDCAuthUnavailable`` are part of the public API: code that
asks for a fresh access token (``tokens.get_access_token``) may receive them and should
let them propagate — the middleware (``process_exception``) and the DRF exception handler
turn them into the HTTP contract (401 ``session_expired`` / 503 ``auth_unavailable``).
"""
from typing import Optional


class OIDCError(Exception):
    """Base class of the library's errors."""


class OIDCLoginRequired(OIDCError):
    """No sign-in where one is required."""

    def __init__(self, reason: str = 'not_authenticated') -> None:
        super().__init__(reason)
        self.reason = reason

    def __str__(self) -> str:
        return self.reason


class OIDCSessionEnded(OIDCError):
    """The sign-in is over (``invalid_grant``, logout, back-channel logout, cap, ...)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

    def __str__(self) -> str:
        return self.reason


class OIDCAuthUnavailable(OIDCError):
    """The OP could not be used right now; the sign-in itself is intact."""

    def __init__(self, category: str, retry_after: Optional[int] = None) -> None:
        super().__init__(category)
        self.category = category
        self.retry_after = retry_after

    def __str__(self) -> str:
        return self.category


class OIDCConfigurationError(OIDCError):
    """The OP or the client is misconfigured (wrong secret, issuer mismatch, ...)."""


class OIDCTokenError(OIDCError):
    """A token failed validation. ``reason`` is a short code, never the token."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

    def __str__(self) -> str:
        return self.reason


#: Deprecated 2.x name, removed in 4.0.
MiddlewareException = OIDCLoginRequired

__all__ = [
    'MiddlewareException',
    'OIDCAuthUnavailable',
    'OIDCConfigurationError',
    'OIDCError',
    'OIDCLoginRequired',
    'OIDCSessionEnded',
    'OIDCTokenError',
]
