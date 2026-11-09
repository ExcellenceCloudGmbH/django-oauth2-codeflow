"""Signals of the session lifecycle.

Receivers must not raise: the library calls them best-effort and logs failures.
"""
from django.dispatch import Signal

#: A sign-in completed. kwargs: ``session`` (OIDCSession), ``request``.
oidc_session_started = Signal()
#: Tokens were refreshed. kwargs: ``session``.
oidc_session_refreshed = Signal()
#: A refresh failed without ending the session (transient or configuration error).
#: kwargs: ``session``, ``category``, ``status``, ``oauth_error``.
oidc_session_refresh_failed = Signal()
#: A sign-in ended. kwargs: ``session``, ``reason``.
oidc_session_ended = Signal()
#: The OP sent a back-channel logout. Fires even when no session matched, so that
#: other holders of the same OP session (e.g. Reflex) can react.
#: kwargs: ``issuer``, ``sid``, ``sub``, ``matched`` (number of sessions ended).
oidc_backchannel_logout_received = Signal()

__all__ = [
    'oidc_backchannel_logout_received',
    'oidc_session_ended',
    'oidc_session_refresh_failed',
    'oidc_session_refreshed',
    'oidc_session_started',
]
