"""Django REST framework integration.

- ``OIDCSessionAuthentication``: ``SessionAuthentication`` (CSRF enforced) that makes DRF
  answer **401** (not 403) to unauthenticated requests, with a ``WWW-Authenticate`` header.
- ``OIDCBearerAuthentication``: the user authenticated by ``BearerAuthMiddleware``.
- ``oidc_exception_handler``: the auth contract for DRF views — adds ``code`` to 401/403
  bodies (``not_authenticated``, ``permission_denied``, ``csrf_failed``) and maps the
  library's ``OIDCSessionEnded`` / ``OIDCAuthUnavailable`` to 401 / 503.

Settings::

    REST_FRAMEWORK = {
        'DEFAULT_AUTHENTICATION_CLASSES': [
            'oauth2_authcodeflow.contrib.drf.OIDCBearerAuthentication',
            'oauth2_authcodeflow.contrib.drf.OIDCSessionAuthentication',
        ],
        'EXCEPTION_HANDLER': 'oauth2_authcodeflow.contrib.drf.oidc_exception_handler',
    }
"""
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)

from rest_framework import (
    exceptions,
    status,
)
from rest_framework.authentication import (
    BaseAuthentication,
    SessionAuthentication,
)
from rest_framework.response import Response

from .. import responses
from ..conf import settings
from ..exceptions import (
    OIDCAuthUnavailable,
    OIDCSessionEnded,
)


class AuthenticateHeaderMixin:
    def authenticate_header(self, request: Any) -> str:
        return f'Session realm="{settings.OIDC_RP_CLIENT_ID}"'


class OIDCSessionAuthentication(AuthenticateHeaderMixin, SessionAuthentication):
    """Session authentication that turns "not authenticated" into 401."""


class OIDCBearerAuthentication(BaseAuthentication):
    """The user authenticated from ``Authorization: Bearer`` by ``BearerAuthMiddleware``."""

    def authenticate(self, request: Any) -> Optional[Tuple[Any, Any]]:
        django_request = getattr(request, '_request', request)
        ctx = getattr(django_request, 'oidc', None)
        user = getattr(django_request, 'user', None)
        if ctx is not None and ctx.is_bearer and user is not None and user.is_authenticated:
            return user, ctx.access_claims
        scheme = request.META.get('HTTP_AUTHORIZATION', '').partition(' ')[0]
        if scheme.lower() == settings.OIDC_AUTHORIZATION_HEADER_PREFIX.lower():
            raise exceptions.AuthenticationFailed('invalid_token')
        return None

    def authenticate_header(self, request: Any) -> str:
        return f'{settings.OIDC_AUTHORIZATION_HEADER_PREFIX} realm="{settings.OIDC_RP_CLIENT_ID}", error="invalid_token"'


def _code(exc: Exception) -> Optional[str]:
    if isinstance(exc, (exceptions.NotAuthenticated, exceptions.AuthenticationFailed)):
        return 'not_authenticated'
    if isinstance(exc, exceptions.PermissionDenied):
        detail = str(getattr(exc, 'detail', ''))
        return 'csrf_failed' if detail.startswith('CSRF Failed') else 'permission_denied'
    return None


def oidc_exception_handler(exc: Exception, context: Dict[str, Any]) -> Optional[Response]:
    request = context.get('request')
    if isinstance(exc, OIDCSessionEnded):
        return Response(
            {'error': 'session_expired', 'login_url': responses.login_path()},
            status=status.HTTP_401_UNAUTHORIZED,
            headers={'WWW-Authenticate': f'Session realm="{settings.OIDC_RP_CLIENT_ID}"', 'Cache-Control': 'no-store'},
        )
    if isinstance(exc, OIDCAuthUnavailable):
        retry = int(exc.retry_after or settings.OIDC_RETRY_AFTER_SECONDS)
        return Response(
            {'error': 'auth_unavailable', 'retry_after': retry},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={'Retry-After': str(retry), 'Cache-Control': 'no-store'},
        )
    # imported lazily: rest_framework.views reads DEFAULT_AUTHENTICATION_CLASSES at import time,
    # which may point back to this module
    from rest_framework.views import exception_handler
    response = exception_handler(exc, context)
    if response is None:
        return None
    code = _code(exc)
    if code and isinstance(response.data, dict):
        response.data.setdefault('code', code)
        if code == 'not_authenticated' and response.status_code == status.HTTP_401_UNAUTHORIZED:
            response.data.setdefault('error', 'not_authenticated')
            response.data.setdefault('login_url', responses.login_path())
    if response.status_code in (401, 403):
        response['Cache-Control'] = 'no-store'
    if request is not None and response.status_code == 401 and 'WWW-Authenticate' not in response:
        response['WWW-Authenticate'] = f'Session realm="{settings.OIDC_RP_CLIENT_ID}"'
    return response


__all__ = [
    'AuthenticateHeaderMixin',
    'OIDCBearerAuthentication',
    'OIDCSessionAuthentication',
    'oidc_exception_handler',
]
