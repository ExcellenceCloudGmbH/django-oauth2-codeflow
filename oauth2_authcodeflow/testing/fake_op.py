"""An in-process OpenID provider for tests."""
import json
import threading
import time
import uuid
from collections import deque
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Union,
)

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

_key_lock = threading.Lock()
_test_key: Optional[rsa.RSAPrivateKey] = None


def test_private_key() -> rsa.RSAPrivateKey:
    """The RSA key of FakeOP and make_signed_token (one per process)."""
    global _test_key
    with _key_lock:
        if _test_key is None:
            _test_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        return _test_key


def make_signed_token(claims: Dict[str, Any], *, key: Any = None, kid: str = 'fake-op-key', alg: str = 'RS256',
                      headers: Optional[Dict[str, Any]] = None) -> str:
    hdrs: Dict[str, Any] = {'kid': kid}
    hdrs.update(headers or {})
    return jwt.encode(claims, key if key is not None else test_private_key(), algorithm=alg, headers=hdrs)


class FakeResponse:
    def __init__(self, status: int, body: Any = None, *, content_type: str = 'application/json', text: Optional[str] = None) -> None:
        self.status_code = status
        self._body = body
        self.headers = {'Content-Type': content_type}
        self.text = text if text is not None else (json.dumps(body) if body is not None else '')

    def json(self) -> Any:
        if self._body is None:
            raise ValueError('no JSON body')
        return self._body


Scripted = Union[Dict[str, Any], tuple, BaseException, Callable[[Dict[str, Any]], Any], FakeResponse]


class _FakeHTTPClient:
    def __init__(self, op: 'FakeOP') -> None:
        self.op = op

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        return self.op._handle(method, url, kwargs)

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        return self.request('GET', url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        return self.request('POST', url, **kwargs)


class FakeOP:
    """Discovery, JWKS, token, userinfo and revocation endpoints served in-process.

    ``install()`` replaces the library's HTTP client and provider cache; use it with
    ``override_settings(**op.django_settings())`` or the ``oidc_op`` pytest fixture.
    Script token-endpoint answers with ``script_token(...)``: a dict (200 JSON), a tuple
    ``(status, body[, content_type])``, an exception instance (raised), a callable
    receiving the form data, or a ``FakeResponse``. Unscripted calls answer like Keycloak.
    """

    def __init__(self, *, issuer: str = 'https://op.example.test/realms/test', access_lifetime: int = 300,
                 refresh_lifetime: int = 1800, kid: str = 'fake-op-key', key: Any = None) -> None:
        self.issuer = issuer
        self.access_lifetime = access_lifetime
        self.refresh_lifetime = refresh_lifetime
        self.kid = kid
        self.key = key if key is not None else test_private_key()
        base = issuer + '/protocol/openid-connect'
        self.discovery_url = issuer + '/.well-known/openid-configuration'
        self.authorization_endpoint = base + '/auth'
        self.token_endpoint = base + '/token'
        self.userinfo_endpoint = base + '/userinfo'
        self.jwks_uri = base + '/certs'
        self.end_session_endpoint = base + '/logout'
        self.revocation_endpoint = base + '/revoke'
        self.discovery_status = 200
        self.lock = threading.Lock()
        self._script: Deque[Scripted] = deque()
        self.token_calls: List[Dict[str, Any]] = []
        self.token_auth: List[Any] = []
        self.userinfo_calls: List[str] = []
        self.revoke_calls: List[Dict[str, Any]] = []
        self.jwks_calls = 0
        self.discovery_calls = 0
        self.codes: Dict[str, Dict[str, Any]] = {}
        self.revoked: set = set()
        self.userinfo_extra: Dict[str, Any] = {}
        self._previous: Optional[tuple] = None

    # --- wiring
    @property
    def client_id(self) -> str:
        from ..conf import settings
        return settings.OIDC_RP_CLIENT_ID

    def django_settings(self) -> Dict[str, Any]:
        return {'OIDC_OP_DISCOVERY_DOCUMENT_URL': self.discovery_url, 'OIDC_OP_ISSUER': self.issuer}

    def install(self) -> 'FakeOP':
        from .. import (
            http,
            provider,
        )
        from ..breaker import reset_breakers
        from ..sessions import reset_refresh_failures
        self._previous = (http._client, provider._provider)
        http.set_client(_FakeHTTPClient(self))  # type: ignore[arg-type]
        provider.set_provider(provider.Provider())
        reset_breakers()
        reset_refresh_failures()
        return self

    def uninstall(self) -> None:
        from .. import (
            http,
            provider,
        )
        from ..breaker import reset_breakers
        from ..sessions import reset_refresh_failures
        if self._previous is not None:
            http.set_client(self._previous[0])
            provider.set_provider(self._previous[1])
            self._previous = None
        reset_breakers()
        reset_refresh_failures()

    def __enter__(self) -> 'FakeOP':
        return self.install()

    def __exit__(self, *exc: Any) -> None:
        self.uninstall()

    # --- documents
    def discovery(self) -> Dict[str, Any]:
        return {
            'issuer': self.issuer,
            'authorization_endpoint': self.authorization_endpoint,
            'token_endpoint': self.token_endpoint,
            'userinfo_endpoint': self.userinfo_endpoint,
            'jwks_uri': self.jwks_uri,
            'end_session_endpoint': self.end_session_endpoint,
            'revocation_endpoint': self.revocation_endpoint,
            'claims_parameter_supported': True,
        }

    def jwks(self) -> Dict[str, Any]:
        jwk = RSAAlgorithm.to_jwk(self.key.public_key(), as_dict=True)
        jwk.update({'kid': self.kid, 'use': 'sig', 'alg': 'RS256'})
        return {'keys': [jwk]}

    def sign(self, claims: Dict[str, Any], **kwargs: Any) -> str:
        return make_signed_token(claims, key=self.key, kid=self.kid, **kwargs)

    # --- tokens
    def issue_tokens(self, sub: str = 'user-1', *, email: Optional[str] = None, sid: Optional[str] = None,
                     nonce: Optional[str] = None, expires_in: Optional[int] = None, refresh: bool = True,
                     client_roles: Optional[List[str]] = None, id_claims: Optional[Dict[str, Any]] = None,
                     access_claims: Optional[Dict[str, Any]] = None, refresh_expires_in: Optional[int] = None) -> Dict[str, Any]:
        now = int(time.time())
        cid = self.client_id
        lifetime = expires_in if expires_in is not None else self.access_lifetime
        email = email or f'{sub}@example.test'
        sid = sid or f'sid-{sub}'
        common = {'iss': self.issuer, 'sub': sub, 'iat': now, 'auth_time': now, 'sid': sid}
        profile = {'email': email, 'email_verified': True, 'given_name': 'Test', 'family_name': 'User',
                   'name': 'Test User', 'preferred_username': email}
        id_c = dict(common, aud=cid, azp=cid, exp=now + max(lifetime, 60), typ='ID', **profile)
        if nonce:
            id_c['nonce'] = nonce
        id_c.update(id_claims or {})
        access_c = dict(common, aud=[cid, 'account'], azp=cid, exp=now + lifetime, typ='Bearer', jti=str(uuid.uuid4()),
                        resource_access={cid: {'roles': list(client_roles or [])}}, scope='openid email profile', **profile)
        access_c.update(access_claims or {})
        data: Dict[str, Any] = {
            'access_token': self.sign(access_c),
            'id_token': self.sign(id_c),
            'expires_in': lifetime,
            'token_type': 'Bearer',
            'session_state': sid,
            'scope': 'openid email profile',
        }
        if refresh:
            rl = refresh_expires_in if refresh_expires_in is not None else self.refresh_lifetime
            data['refresh_token'] = self.sign({'iss': self.issuer, 'sub': sub, 'sid': sid, 'aud': self.issuer,
                                               'iat': now, 'exp': now + rl, 'typ': 'Refresh', 'jti': str(uuid.uuid4())})
            data['refresh_expires_in'] = rl
        return data

    def authorization_code(self, sub: str = 'user-1', *, nonce: Optional[str] = None, **kwargs: Any) -> str:
        code = 'code-' + uuid.uuid4().hex
        self.codes[code] = {'sub': sub, 'nonce': nonce, 'kwargs': kwargs}
        return code

    def script_token(self, *answers: Scripted) -> None:
        with self.lock:
            self._script.extend(answers)

    def refresh_calls(self) -> int:
        with self.lock:
            return sum(1 for call in self.token_calls if call.get('grant_type') == 'refresh_token')

    # --- request handling
    def _handle(self, method: str, url: str, kwargs: Dict[str, Any]) -> FakeResponse:
        path = url.split('?', 1)[0]
        if path == self.discovery_url:
            with self.lock:
                self.discovery_calls += 1
            if self.discovery_status != 200:
                return FakeResponse(self.discovery_status, None, content_type='text/html', text='<html>unavailable</html>')
            return FakeResponse(200, self.discovery())
        if path == self.jwks_uri:
            with self.lock:
                self.jwks_calls += 1
            return FakeResponse(200, self.jwks())
        if path == self.token_endpoint:
            with self.lock:
                self.token_auth.append(kwargs.get('auth'))
            return self._token(dict(kwargs.get('data') or {}))
        if path == self.userinfo_endpoint:
            auth = (kwargs.get('headers') or {}).get('Authorization', '')
            token = auth.partition(' ')[2]
            with self.lock:
                self.userinfo_calls.append(token)
            claims = jwt.decode(token, options={'verify_signature': False}) if token else {}
            profile = ('sub', 'email', 'email_verified', 'name', 'given_name', 'family_name', 'preferred_username')
            body: Dict[str, Any] = {k: claims[k] for k in profile if k in claims}
            body.update(self.userinfo_extra)
            return FakeResponse(200, body)
        if path == self.revocation_endpoint:
            data = dict(kwargs.get('data') or {})
            with self.lock:
                self.revoke_calls.append(data)
                self.revoked.add(data.get('token'))
            return FakeResponse(200, None, text='')
        return FakeResponse(404, None, content_type='text/plain', text='not found')

    def _token(self, data: Dict[str, Any]) -> FakeResponse:
        with self.lock:
            self.token_calls.append(data)
            scripted = self._script.popleft() if self._script else None
        if scripted is not None:
            if isinstance(scripted, BaseException):
                raise scripted
            if callable(scripted) and not isinstance(scripted, FakeResponse):
                scripted = scripted(data)
            if isinstance(scripted, FakeResponse):
                return scripted
            if isinstance(scripted, dict):
                return FakeResponse(200, scripted)
            status, body, *rest = scripted
            if isinstance(body, dict):
                return FakeResponse(status, body, content_type=rest[0] if rest else 'application/json')
            return FakeResponse(status, None, content_type=rest[0] if rest else 'text/html', text=str(body))
        grant = data.get('grant_type')
        if grant == 'authorization_code':
            entry = self.codes.pop(data.get('code', ''), None)
            if entry is None:
                return FakeResponse(400, {'error': 'invalid_grant', 'error_description': 'Code not valid'})
            return FakeResponse(200, self.issue_tokens(entry['sub'], nonce=entry['nonce'], **entry['kwargs']))
        if grant == 'refresh_token':
            token = data.get('refresh_token', '')
            if token in self.revoked:
                return FakeResponse(400, {'error': 'invalid_grant', 'error_description': 'Token is not active'})
            try:
                claims = jwt.decode(token, options={'verify_signature': False})
            except jwt.PyJWTError:
                return FakeResponse(400, {'error': 'invalid_grant', 'error_description': 'Invalid refresh token'})
            if claims.get('exp', 0) < time.time():
                return FakeResponse(400, {'error': 'invalid_grant', 'error_description': 'Token is not active'})
            return FakeResponse(200, self.issue_tokens(claims.get('sub', 'user-1'), sid=claims.get('sid')))
        return FakeResponse(400, {'error': 'unsupported_grant_type'})

    # --- back-channel logout
    def logout_token(self, *, sub: Optional[str] = None, sid: Optional[str] = None, iat: Optional[int] = None,
                     jti: Optional[str] = None, extra: Optional[Dict[str, Any]] = None, events: bool = True) -> str:
        claims: Dict[str, Any] = {'iss': self.issuer, 'aud': self.client_id, 'iat': iat if iat is not None else int(time.time()),
                                  'jti': jti or str(uuid.uuid4())}
        if events:
            claims['events'] = {'http://schemas.openid.net/event/backchannel-logout': {}}
        if sub:
            claims['sub'] = sub
        if sid:
            claims['sid'] = sid
        claims.update(extra or {})
        return self.sign(claims, headers={'typ': 'logout+jwt'})


__all__ = ['FakeOP', 'FakeResponse', 'make_signed_token', 'test_private_key']
