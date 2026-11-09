"""Django users from OIDC claims.

- A user is created only when ``OIDC_CREATE_USER`` is on; it gets an unusable password.
- An existing user's password is never touched (unless ``OIDC_UNUSABLE_PASSWORD`` asks for
  it): changing it would end every other Django session of that user.
- Fields are saved only when a claim changed them.
- An inactive existing user stays inactive (2.x silently re-activated them at login).
"""
from inspect import signature
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from django.contrib.auth import get_user_model
from django.contrib.auth.models import AbstractBaseUser
from django.http import HttpRequest

from .conf import settings
from .exceptions import OIDCTokenError


def validate_claims(claims: Dict[str, Any]) -> None:
    expected = [settings.OIDC_OP_EXPECTED_EMAIL_CLAIM] + list(settings.OIDC_OP_EXPECTED_CLAIMS)
    for name in expected:
        if name and name not in claims:
            raise OIDCTokenError('missing_claims')


def _claim(spec: Any, claims: Dict[str, Any], default_name: Optional[str] = None) -> str:
    if callable(spec):
        return spec(claims) or ''
    name = spec or default_name
    return str(claims.get(name, '') or '') if name else ''


def _snapshot(user: AbstractBaseUser) -> Dict[str, Any]:
    return {f.attname: getattr(user, f.attname) for f in user._meta.concrete_fields}


def provision_user(claims: Dict[str, Any], *, request: Optional[HttpRequest] = None, access_token: str = '') -> Optional[AbstractBaseUser]:
    """The Django user for these claims (created when allowed); None when not allowed."""
    validate_claims(claims)
    User = get_user_model()
    username = settings.OIDC_DJANGO_USERNAME_FUNC(claims)
    lookup = {User.USERNAME_FIELD: username}
    try:
        user = User._default_manager.get(**lookup)
        created = False
    except User.DoesNotExist:
        if not settings.OIDC_CREATE_USER:
            return None
        user = User(**lookup)
        created = True
    before = None if created else _snapshot(user)

    if hasattr(user, 'email'):
        user.email = _claim(settings.OIDC_EMAIL_CLAIM, claims, settings.OIDC_OP_EXPECTED_EMAIL_CLAIM)
    if hasattr(user, 'first_name'):
        user.first_name = _claim(settings.OIDC_FIRSTNAME_CLAIM, claims)
    if hasattr(user, 'last_name'):
        user.last_name = _claim(settings.OIDC_LASTNAME_CLAIM, claims)
    if created:
        user.set_unusable_password()
        user.is_active = True
    elif settings.OIDC_UNUSABLE_PASSWORD:
        user.set_unusable_password()
    extend = settings.OIDC_EXTEND_USER
    if callable(extend):
        if len(signature(extend).parameters) > 2:
            extend(user, claims, request, access_token)
        else:  # backward compatibility
            extend(user, claims)

    if created:
        user.save()
    else:
        after = _snapshot(user)
        changed: List[str] = [name for name, value in after.items() if before is not None and before.get(name) != value]
        if changed:
            fields = [f.name for f in user._meta.concrete_fields if f.attname in changed and not f.primary_key]
            user.save(update_fields=fields)
    return user


__all__ = ['provision_user', 'validate_claims']
