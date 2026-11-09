"""Public access to the tokens of the current request.

>>> from oauth2_authcodeflow.tokens import get_access_token
>>> token = get_access_token(request, min_ttl=60)   # valid for at least 60 s, refreshed if needed

Raises ``OIDCSessionEnded`` (the sign-in is over) or ``OIDCAuthUnavailable`` (the OP is
unreachable and the stored token is too old); let them propagate, the middleware and the
DRF exception handler answer per the auth contract.
"""
from typing import (
    Any,
    Dict,
    Optional,
)

from django.http import HttpRequest

from .crypto import TokenDecryptError
from .exceptions import OIDCSessionEnded
from .sessions import (
    end_session,
    ensure_fresh,
)


def get_access_token(request: HttpRequest, *, min_ttl: int = 0) -> Optional[str]:
    """The access token of the request's sign-in (None without an OIDC sign-in)."""
    ctx = getattr(request, 'oidc', None)
    row = None
    if ctx is not None:
        if ctx.is_bearer:
            return ctx._bearer_token
        row = ctx._row
    if row is None:
        # inside the login callback (e.g. user_logged_in receivers), before request.oidc exists
        row = getattr(request, '_oidc_row', None)
    if row is None:
        return None
    fresh = ensure_fresh(row, min_ttl=min_ttl, wait=True)
    if ctx is not None and fresh is not row:
        ctx._update(fresh)
    try:
        return fresh.access_token
    except TokenDecryptError:
        end_session(fresh, 'decrypt_failed')
        raise OIDCSessionEnded('decrypt_failed')


def get_id_claims(request: HttpRequest) -> Dict[str, Any]:
    """The allow-listed profile claims of the request's sign-in ({} without one)."""
    ctx = getattr(request, 'oidc', None)
    return dict(ctx.claims) if ctx is not None else {}


__all__ = ['get_access_token', 'get_id_claims']
