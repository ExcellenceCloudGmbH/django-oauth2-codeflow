"""Authentication backends.

- ``AuthenticationBackend`` marks Django sessions created by the OIDC callback (the code
  exchange itself lives in ``views.CallbackView``). Keep it in ``AUTHENTICATION_BACKENDS``.
- ``BearerAuthenticationBackend`` validates **access tokens** (never id tokens): signature,
  issuer, ``typ``, audience, lifetime; asymmetric algorithms only.
"""
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)

from django.contrib.auth.backends import ModelBackend
from django.contrib.auth.models import AbstractBaseUser
from django.http import HttpRequest

from .conf import settings
from .exceptions import OIDCTokenError
from .jwt_validation import validate_access_token
from .users import provision_user


class AuthenticationBackend(ModelBackend):
    """Backend of OIDC sign-ins. ``authenticate()`` returns None: users sign in through the
    callback view, which calls ``django.contrib.auth.login`` with this backend."""

    def authenticate(self, request: Optional[HttpRequest], username: Optional[str] = None,
                     password: Optional[str] = None, **kwargs: Any) -> Optional[AbstractBaseUser]:
        return None


class BearerAuthenticationBackend(ModelBackend):
    def authenticate(self, request: Optional[HttpRequest], username: Optional[str] = None,
                     password: Optional[str] = None, **kwargs: Any) -> Optional[AbstractBaseUser]:
        if request is None:
            return None
        scheme, _, token = request.headers.get('Authorization', '').partition(' ')
        if scheme.lower() != settings.OIDC_AUTHORIZATION_HEADER_PREFIX.lower() or not token.strip():
            return None
        try:
            user, _claims = self.authenticate_token(token.strip(), request=request)
        except OIDCTokenError:
            return None
        return user

    def authenticate_token(self, token: str, request: Optional[HttpRequest] = None) -> Tuple[AbstractBaseUser, Dict[str, Any]]:
        """(user, claims) for a valid access token; raises OIDCTokenError otherwise."""
        claims = validate_access_token(token)
        user = provision_user(claims, request=request, access_token=token)
        if user is None:
            raise OIDCTokenError('user_not_allowed')
        if not getattr(user, 'is_active', True):
            raise OIDCTokenError('user_inactive')
        return user, claims


__all__ = ['AuthenticationBackend', 'BearerAuthenticationBackend']
