"""Middlewares.

Order (system check E001)::

    SessionMiddleware → AuthenticationMiddleware → BearerAuthMiddleware
        → OIDCSessionMiddleware → … → LoginRequiredMiddleware

- ``OIDCSessionMiddleware`` loads the request's sign-in, keeps its access token fresh,
  slides the Django session and sets ``request.oidc``. It never redirects anonymous
  requests and never flushes a session that has no OIDC sign-in.
- ``LoginRequiredMiddleware`` is policy only: anonymous requests to non-exempt paths get
  the contract's "not authenticated" answer (``check_login_required`` can be overridden).
- ``BearerAuthMiddleware`` authenticates ``Authorization: Bearer <access token>`` requests
  without creating a session.
"""
import re
import warnings
from functools import lru_cache
from typing import (
    Callable,
    Optional,
    Pattern,
    Tuple,
)

from django.http import (
    HttpRequest,
    HttpResponse,
)
from django.urls import (
    NoReverseMatch,
    reverse,
)

from . import (
    constants,
    responses,
    sessions,
)
from .conf import settings
from .context import OIDCContext
from .exceptions import (
    OIDCAuthUnavailable,
    OIDCLoginRequired,
    OIDCSessionEnded,
    OIDCTokenError,
)

GetResponseCallable = Callable[[HttpRequest], HttpResponse]

_OIDC_URL_NAMES = (
    constants.OIDC_URL_AUTHENTICATION_NAME,
    constants.OIDC_URL_CALLBACK_NAME,
    constants.OIDC_URL_LOGOUT_NAME,
    constants.OIDC_URL_TOTAL_LOGOUT_NAME,
    constants.OIDC_URL_BACKCHANNEL_LOGOUT_NAME,
)


def oidc_endpoint_patterns() -> Tuple[str, ...]:
    patterns = []
    for name in _OIDC_URL_NAMES:
        try:
            patterns.append('^' + re.escape(reverse(name)))
        except NoReverseMatch:
            continue
    return tuple(patterns)


@lru_cache(maxsize=16)
def _compile(patterns: Tuple[str, ...]) -> Tuple[Pattern, ...]:
    return tuple(re.compile(p) for p in patterns)


def _matches(patterns: Tuple[str, ...], path: str) -> bool:
    return any(p.search(path) for p in _compile(patterns))


class OIDCSessionMiddleware:
    sync_capable = True
    async_capable = False

    def __init__(self, get_response: GetResponseCallable) -> None:
        self.get_response = get_response

    def exempt_patterns(self) -> Tuple[str, ...]:
        return oidc_endpoint_patterns() + tuple(str(p) for p in settings.OIDC_SESSION_EXEMPT_URL_PATTERNS)

    def __call__(self, request: HttpRequest) -> HttpResponse:
        if getattr(request, '_oidc_mw_done', False):  # aliases listed next to it run once
            return self.get_response(request)
        request._oidc_mw_done = True  # type: ignore[attr-defined]
        ctx = getattr(request, 'oidc', None)
        if ctx is not None and ctx.is_bearer:
            return self.get_response(request)
        request.oidc = None  # type: ignore[attr-defined]
        if _matches(self.exempt_patterns(), request.path_info):
            return self.get_response(request)
        try:
            row = sessions.load_for_request(request)
            if row is not None:
                if settings.OIDC_REFRESH_MODE != 'lazy':
                    row = sessions.ensure_fresh(row)
                request.oidc = OIDCContext.from_row(row)  # type: ignore[attr-defined]
                sessions.renew_django_session(request, row)
        except OIDCSessionEnded:
            sessions.logout_request(request, 'session_expired')
            return responses.unauthenticated(request, 'session_expired')
        except OIDCAuthUnavailable as e:
            return responses.auth_unavailable(request, e)
        return self.get_response(request)

    def process_exception(self, request: HttpRequest, exception: Exception) -> Optional[HttpResponse]:
        """Map the library's exceptions raised in views (``get_access_token``) to the contract."""
        if isinstance(exception, OIDCSessionEnded):
            sessions.logout_request(request, exception.reason)
            return responses.unauthenticated(request, 'session_expired')
        if isinstance(exception, OIDCAuthUnavailable):
            return responses.auth_unavailable(request, exception)
        if isinstance(exception, OIDCLoginRequired):
            return responses.unauthenticated(request, exception.reason)
        return None


class LoginRequiredMiddleware:
    """Anonymous requests to non-exempt paths get the contract's "not authenticated" answer:
    a redirect to the login view for navigations, a JSON 401 for everything else."""
    sync_capable = True
    async_capable = False

    def __init__(self, get_response: GetResponseCallable) -> None:
        self.get_response = get_response

    @property
    def exempt_urls(self) -> Tuple[str, ...]:
        return oidc_endpoint_patterns() + tuple(str(p) for p in settings.OIDC_LOGIN_REQUIRED_EXEMPT_URL_PATTERNS)

    def is_login_required_for_url(self, request: HttpRequest) -> bool:
        return not _matches(self.exempt_urls, request.path_info)

    def check_login_required(self, request: HttpRequest) -> None:
        """Raise OIDCLoginRequired when this request needs a signed-in user. Override hook."""
        user = getattr(request, 'user', None)
        if user is not None and user.is_authenticated:
            return
        if not self.is_login_required_for_url(request):
            return
        raise OIDCLoginRequired('not_authenticated')

    def __call__(self, request: HttpRequest) -> HttpResponse:
        try:
            self.check_login_required(request)
        except OIDCLoginRequired as e:
            return responses.unauthenticated(request, e.reason)
        return self.get_response(request)


class BearerAuthMiddleware:
    """Authenticate ``Authorization: Bearer <access token>``. Never creates a session.
    Other schemes (``Api-Key …``, ``Basic …``) are left alone; an invalid bearer token leaves
    the request anonymous (later checks answer 401 with ``error="invalid_token"``)."""
    sync_capable = True
    async_capable = False

    def __init__(self, get_response: GetResponseCallable) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        header = request.headers.get('Authorization', '')
        scheme, _, token = header.partition(' ')
        token = token.strip()
        if not token or scheme.lower() != settings.OIDC_AUTHORIZATION_HEADER_PREFIX.lower():
            return self.get_response(request)
        from .auth import BearerAuthenticationBackend
        try:
            user, claims = BearerAuthenticationBackend().authenticate_token(token)
        except OIDCTokenError as e:
            request._oidc_bearer_error = e.reason  # type: ignore[attr-defined]
            return self.get_response(request)
        except OIDCAuthUnavailable as e:
            return responses.auth_unavailable(request, e)
        request.user = user  # type: ignore[assignment]
        request._cached_user = user  # type: ignore[attr-defined]
        request.oidc = OIDCContext.from_bearer(token, claims)  # type: ignore[attr-defined]
        return self.get_response(request)


def _deprecated(name: str) -> None:
    warnings.warn(
        f"oauth2_authcodeflow.middleware.{name} is deprecated: use OIDCSessionMiddleware (removed in 4.0)",
        DeprecationWarning,
        stacklevel=3,
    )


class RefreshAccessTokenMiddleware(OIDCSessionMiddleware):
    """Deprecated alias of OIDCSessionMiddleware (runs once per request with it)."""

    def __init__(self, get_response: GetResponseCallable) -> None:
        _deprecated(type(self).__name__)
        super().__init__(get_response)


class RefreshSessionMiddleware(OIDCSessionMiddleware):
    """Deprecated alias of OIDCSessionMiddleware (runs once per request with it)."""

    def __init__(self, get_response: GetResponseCallable) -> None:
        _deprecated(type(self).__name__)
        super().__init__(get_response)


class Oauth2MiddlewareMixin:
    """Deprecated 2.x base class. ``check_function(request)`` raising ``MiddlewareException``
    (``OIDCLoginRequired``) yields the contract's "not authenticated" answer."""

    def __init__(self, get_response: GetResponseCallable, token_type: Optional[str] = None,
                 check_function: Optional[Callable[[HttpRequest], None]] = None) -> None:
        _deprecated(type(self).__name__)
        self.get_response = get_response
        self.token_type = token_type
        self.check_function = check_function

    def __call__(self, request: HttpRequest) -> HttpResponse:
        if self.check_function:
            try:
                self.check_function(request)
            except OIDCLoginRequired as e:
                return responses.unauthenticated(request, e.reason)
        return self.get_response(request)


__all__ = [
    'BearerAuthMiddleware',
    'LoginRequiredMiddleware',
    'OIDCSessionMiddleware',
    'Oauth2MiddlewareMixin',
    'RefreshAccessTokenMiddleware',
    'RefreshSessionMiddleware',
    'oidc_endpoint_patterns',
]
