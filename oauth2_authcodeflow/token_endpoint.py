"""Calls to the OP token, userinfo and revocation endpoints, and their classification.

Classification is done on the OAuth ``error`` field, never on the status code alone:

=====================================================  ==========  ===================
token endpoint result                                  category    sign-in
=====================================================  ==========  ===================
200 JSON with ``access_token``                         ok          updated
``error=invalid_grant``                                terminal    ended
``invalid_grant`` mentioning the issuer                config      kept
``invalid_client``, ``unauthorized_client``,           config      kept
``unsupported_grant_type``, ``invalid_scope``,
``invalid_request``
429, 5xx, HTML (e.g. a proxy/WAF page), 401/403        transient   kept
without an OAuth error, 2xx without ``access_token``,
timeouts, connection or TLS errors
=====================================================  ==========  ===================
"""
import logging
import time
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)
from urllib.parse import quote

from requests import RequestException

from . import events
from .conf import settings
from .exceptions import (
    OIDCAuthUnavailable,
    OIDCConfigurationError,
)
from .http import get_client
from .provider import get_provider

OK = 'ok'
TERMINAL = 'terminal'
CONFIG = 'config'
TRANSIENT = 'transient'

CONFIG_ERRORS = frozenset({'invalid_client', 'unauthorized_client', 'unsupported_grant_type', 'invalid_scope', 'invalid_request'})
TRANSIENT_ERRORS = frozenset({'temporarily_unavailable', 'server_error', 'slow_down'})


@dataclass
class TokenOutcome:
    category: str
    data: Dict[str, Any] = field(default_factory=dict)
    status: Optional[int] = None
    oauth_error: Optional[str] = None
    description: str = ''
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.category == OK


def _issuer_problem(description: str) -> bool:
    text = description.lower()
    return 'issuer' in text or 'iss ' in text


def classify(status: int, body: Any, content_type: str = '') -> Tuple[str, Optional[str], str]:
    """(category, oauth_error, description) of a token-endpoint response."""
    data = body if isinstance(body, dict) else None
    if data is not None and data.get('error'):
        error = str(data.get('error'))
        description = str(data.get('error_description') or '')
        if error == 'invalid_grant':
            return (CONFIG if _issuer_problem(description) else TERMINAL), error, description
        if error in CONFIG_ERRORS:
            return CONFIG, error, description
        return TRANSIENT, error, description
    if status == 200 and data is not None and data.get('access_token'):
        return OK, None, ''
    if status == 200:
        return TRANSIENT, None, 'no access_token in a 2xx response'
    return TRANSIENT, None, f'HTTP {status}' + (f' ({content_type.split(";")[0]})' if content_type else '')


def _client_auth(data: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    client_id = settings.OIDC_RP_CLIENT_ID
    secret = settings.OIDC_RP_CLIENT_SECRET or ''
    method = settings.OIDC_RP_CLIENT_AUTH_METHOD or ('client_secret_post' if secret else 'none')
    data['client_id'] = client_id
    if method == 'client_secret_post':
        data['client_secret'] = secret
        return None
    if method == 'client_secret_basic':
        # RFC 6749 §2.3.1: form-urlencode id and secret before Basic encoding
        return quote(client_id, safe=''), quote(secret, safe='')
    return None


def _post(url: str, data: Dict[str, Any], what: str) -> TokenOutcome:
    auth = _client_auth(data)
    headers = {'Accept': 'application/json'}
    if settings.OIDC_RP_AZURE_SPA and 'redirect_uri' in data:
        headers['Origin'] = data['redirect_uri']
    started = time.monotonic()
    try:
        resp = get_client().post(url, data=data, headers=headers, auth=auth)
    except RequestException as e:
        return TokenOutcome(TRANSIENT, status=None, description=type(e).__name__,
                            duration_ms=int((time.monotonic() - started) * 1000))
    duration = int((time.monotonic() - started) * 1000)
    content_type = resp.headers.get('Content-Type', '')
    try:
        body = resp.json()
    except ValueError:
        body = None
    category, error, description = classify(resp.status_code, body, content_type)
    return TokenOutcome(
        category,
        data=body if category == OK and isinstance(body, dict) else {},
        status=resp.status_code,
        oauth_error=error,
        description=events.redact(description) or '',
        duration_ms=duration,
    )


def _token_endpoint() -> str:
    return get_provider().metadata().token_endpoint


def exchange_code(code: str, redirect_uri: str, code_verifier: Optional[str]) -> TokenOutcome:
    data: Dict[str, Any] = {'grant_type': 'authorization_code', 'code': code, 'redirect_uri': redirect_uri}
    if code_verifier:
        data['code_verifier'] = code_verifier
    return _post(_token_endpoint(), data, 'code')


def refresh(refresh_token: str) -> TokenOutcome:
    try:
        url = _token_endpoint()
    except OIDCAuthUnavailable as e:
        return TokenOutcome(TRANSIENT, description=e.category)
    except OIDCConfigurationError as e:
        return TokenOutcome(CONFIG, description=str(e))
    return _post(url, {'grant_type': 'refresh_token', 'refresh_token': refresh_token}, 'refresh')


def revoke(token: str, token_type_hint: str = 'refresh_token') -> bool:
    """RFC 7009 revocation. Best effort: False when not possible or not accepted."""
    try:
        url = get_provider().metadata().revocation_endpoint
    except Exception:
        return False
    if not url:
        return False
    outcome = _post(url, {'token': token, 'token_type_hint': token_type_hint}, 'revoke')
    accepted = outcome.status == 200
    if not accepted:
        events.emit('oidc.revoke.failed', logging.WARNING, status=outcome.status, error=outcome.oauth_error)
    return accepted


def fetch_userinfo(access_token: str) -> Optional[Dict[str, Any]]:
    """Userinfo claims, or None when unavailable (callers fall back to the id token)."""
    try:
        url = get_provider().metadata().userinfo_endpoint
    except Exception:
        return None
    if not url:
        return None
    try:
        resp = get_client().get(url, headers={
            'Authorization': f'{settings.OIDC_AUTHORIZATION_HEADER_PREFIX} {access_token}',
            'Accept': 'application/json',
        })
    except RequestException:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


__all__ = [
    'CONFIG',
    'OK',
    'TERMINAL',
    'TRANSIENT',
    'TokenOutcome',
    'classify',
    'exchange_code',
    'fetch_userinfo',
    'refresh',
    'revoke',
]
