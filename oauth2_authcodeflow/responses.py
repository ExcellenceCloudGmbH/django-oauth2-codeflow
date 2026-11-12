"""The HTTP auth contract (v2).

=============================  ===========================  ==========================  ===========================================
case                           navigation                   frame navigation            background request (fetch/XHR, API, bearer)
=============================  ===========================  ==========================  ===========================================
no session / session ended     302 login view (``next``)    401 HTML, ``_top`` link     401 JSON ``not_authenticated`` /
                                                                                        ``session_expired`` + ``login_url``
auth backend unavailable       503 HTML, auto-retry         503 HTML                    503 JSON ``auth_unavailable`` + Retry-After
=============================  ===========================  ==========================  ===========================================

All responses carry ``Cache-Control: no-store``; JSON 401s carry ``WWW-Authenticate``.
"""
import re
from functools import lru_cache
from html import escape
from typing import (
    Optional,
    Pattern,
    Tuple,
)
from urllib.parse import urlencode

from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseRedirect,
    JsonResponse,
)
from django.urls import reverse

from . import constants
from .conf import settings
from .exceptions import OIDCAuthUnavailable

NAVIGATION = 'navigation'
FRAME = 'frame'
API = 'api'

_FRAME_DESTINATIONS = frozenset({'iframe', 'frame', 'embed', 'object'})


@lru_cache(maxsize=8)
def _compile(patterns: Tuple[str, ...]) -> Tuple[Pattern, ...]:
    return tuple(re.compile(p) for p in patterns)


def api_patterns() -> Tuple[Pattern, ...]:
    return _compile(tuple(str(p) for p in settings.OIDC_API_URL_PATTERNS))


def request_kind(request: HttpRequest) -> str:
    """NAVIGATION, FRAME or API for a request."""
    if request.method not in ('GET', 'HEAD'):
        return API
    if any(p.search(request.path_info) for p in api_patterns()):
        return API
    if 'HTTP_AUTHORIZATION' in request.META:
        return API
    mode = request.headers.get('Sec-Fetch-Mode')
    if mode:
        if mode != 'navigate':
            return API
        return FRAME if request.headers.get('Sec-Fetch-Dest') in _FRAME_DESTINATIONS else NAVIGATION
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return API
    return NAVIGATION if 'text/html' in request.headers.get('Accept', '') else API


def login_path() -> str:
    return settings.OIDC_LOGIN_URL or reverse(constants.OIDC_URL_AUTHENTICATION_NAME)


def login_url(next_url: Optional[str] = None) -> str:
    base = login_path()
    if not next_url:
        return base
    return f'{base}?{urlencode({settings.OIDC_REDIRECT_OK_FIELD_NAME: next_url})}'


def _no_store(response: HttpResponse) -> HttpResponse:
    response['Cache-Control'] = 'no-store'
    response['Pragma'] = 'no-cache'
    return response


def www_authenticate(request: HttpRequest) -> str:
    scheme = request.META.get('HTTP_AUTHORIZATION', '').partition(' ')[0].lower()
    if getattr(request, '_oidc_bearer_error', None) or scheme == settings.OIDC_AUTHORIZATION_HEADER_PREFIX.lower():
        return f'{settings.OIDC_AUTHORIZATION_HEADER_PREFIX} error="invalid_token"'
    return f'Session realm="{settings.OIDC_RP_CLIENT_ID}"'


def unauthenticated(request: HttpRequest, reason: str = 'not_authenticated') -> HttpResponse:
    """The contract's answer when a request has no (more) sign-in."""
    if reason not in ('not_authenticated', 'session_expired'):
        reason = 'not_authenticated'
    kind = request_kind(request)
    next_url = request.get_full_path()
    if kind == NAVIGATION and settings.OIDC_NAVIGATION_REDIRECT:
        return _no_store(HttpResponseRedirect(login_url(next_url)))
    if kind == FRAME:
        url = escape(login_url(next_url))
        body = (
            '<!doctype html><html><head><meta charset="utf-8"><title>Sign-in required</title></head>'
            '<body style="font-family:sans-serif;margin:2rem">'
            '<p>Your session has ended.</p>'
            f'<p><a href="{url}" target="_top" rel="noopener">Sign in again</a></p>'
            '</body></html>'
        )
        response = HttpResponse(body, status=401, content_type='text/html; charset=utf-8')
        response['WWW-Authenticate'] = www_authenticate(request)
        return _no_store(response)
    response = JsonResponse({'error': reason, 'login_url': login_path()}, status=401)
    response['WWW-Authenticate'] = www_authenticate(request)
    return _no_store(response)


def auth_unavailable(request: HttpRequest, exc: Optional[OIDCAuthUnavailable] = None) -> HttpResponse:
    """The contract's answer when the OP cannot be used right now (the sign-in is intact)."""
    retry = int((exc.retry_after if exc and exc.retry_after else None) or settings.OIDC_RETRY_AFTER_SECONDS)
    if request_kind(request) == API:
        response: HttpResponse = JsonResponse({'error': 'auth_unavailable', 'retry_after': retry}, status=503)
    else:
        body = (
            '<!doctype html><html><head><meta charset="utf-8">'
            f'<meta http-equiv="refresh" content="{retry}"><title>Temporarily unavailable</title></head>'
            '<body style="font-family:sans-serif;margin:2rem">'
            '<p>Sign-in is temporarily unavailable. This page retries automatically.</p>'
            '</body></html>'
        )
        response = HttpResponse(body, status=503, content_type='text/html; charset=utf-8')
    response['Retry-After'] = str(retry)
    return _no_store(response)


__all__ = [
    'API',
    'FRAME',
    'NAVIGATION',
    'api_patterns',
    'auth_unavailable',
    'login_path',
    'login_url',
    'request_kind',
    'unauthenticated',
    'www_authenticate',
]
