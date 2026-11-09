import time
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from django.contrib.auth import (
    BACKEND_SESSION_KEY,
    HASH_SESSION_KEY,
    SESSION_KEY,
)
from django.http import HttpRequest

from .. import (
    constants,
    sessions,
)
from ..conf import settings
from ..context import OIDCContext
from ..models import OIDCSession
from .fake_op import make_signed_token

DEFAULT_TEST_ISSUER = 'https://op.example.test/realms/test'


def create_oidc_session(client_or_request: Any, user: Any, *, client_roles: Optional[List[str]] = None,
                        claims: Optional[Dict[str, Any]] = None, access_expires_in: int = 300, refresh: bool = True,
                        issuer: Optional[str] = None, sid: Optional[str] = None, **access_claims: Any) -> OIDCSession:
    """Sign ``user`` in through OIDC without an OP.

    ``client_or_request``: a ``django.test.Client`` (cookies are set on it) or an
    ``HttpRequest`` (its session gets the sign-in and ``request.oidc`` is set). Tokens are
    RS256-signed with the FakeOP test key. ``oauth2_authcodeflow.auth.AuthenticationBackend``
    must be in ``AUTHENTICATION_BACKENDS``.
    """
    issuer = issuer or DEFAULT_TEST_ISSUER
    claims = dict(claims or {})
    sub = str(claims.get('sub') or f'sub-{user.pk}')
    sid = sid or str(claims.get('sid') or f'sid-{sub}')
    now = int(time.time())
    cid = settings.OIDC_RP_CLIENT_ID
    email = getattr(user, 'email', '') or f'{sub}@example.test'
    id_claims: Dict[str, Any] = {'iss': issuer, 'sub': sub, 'aud': cid, 'azp': cid, 'iat': now,
                                 'exp': now + max(access_expires_in, 60), 'sid': sid, 'email': email, 'typ': 'ID'}
    id_claims.update(claims)
    access: Dict[str, Any] = {'iss': issuer, 'sub': sub, 'aud': [cid], 'azp': cid, 'iat': now, 'exp': now + access_expires_in,
                              'typ': 'Bearer', 'sid': sid, 'email': email, 'resource_access': {cid: {'roles': list(client_roles or [])}}}
    access.update(access_claims)
    data: Dict[str, Any] = {'access_token': make_signed_token(access), 'id_token': make_signed_token(id_claims),
                            'expires_in': access_expires_in}
    if refresh:
        data['refresh_token'] = make_signed_token({'iss': issuer, 'sub': sub, 'sid': sid, 'iat': now, 'exp': now + 1800, 'typ': 'Refresh'})
        data['refresh_expires_in'] = 1800
    row = sessions.create_session(user=user, token_data=data, id_claims=id_claims)

    if isinstance(client_or_request, HttpRequest):
        request = client_or_request
        if not hasattr(request, 'session'):
            from importlib import import_module

            from django.conf import settings as dj_settings
            request.session = import_module(dj_settings.SESSION_ENGINE).SessionStore()
        request.user = user
        request.session[SESSION_KEY] = user._meta.pk.value_to_string(user)
        request.session[BACKEND_SESSION_KEY] = constants.AUTH_BACKEND_PATH
        request.session[HASH_SESSION_KEY] = user.get_session_auth_hash() if hasattr(user, 'get_session_auth_hash') else ''
        request.session[settings.OIDC_SESSION_KEY] = str(row.pk)
        request.session[constants.SESSION_RENEWED_AT] = now
        request.session.save()
        row.django_session_key = request.session.session_key or ''
        request.oidc = OIDCContext.from_row(row)  # type: ignore[attr-defined]
    else:
        client = client_or_request
        client.force_login(user, backend=constants.AUTH_BACKEND_PATH)
        session = client.session
        session[settings.OIDC_SESSION_KEY] = str(row.pk)
        session[constants.SESSION_RENEWED_AT] = now
        session.save()
        row.django_session_key = session.session_key or ''
    OIDCSession.objects.filter(pk=row.pk).update(django_session_key=row.django_session_key)
    return row


__all__ = ['DEFAULT_TEST_ISSUER', 'create_oidc_session']
