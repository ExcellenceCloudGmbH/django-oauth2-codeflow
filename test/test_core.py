"""Settings, encryption, circuit breaker, keyed locks, token-endpoint classification, provider and JWT validation."""
import threading
import time
import warnings
from base64 import urlsafe_b64encode

import jwt
import pytest
import requests
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings

from oauth2_authcodeflow import token_endpoint
from oauth2_authcodeflow.breaker import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    CircuitBreaker,
)
from oauth2_authcodeflow.conf import (
    Settings,
    settings,
)
from oauth2_authcodeflow.crypto import (
    TokenCipher,
    TokenDecryptError,
    get_cipher,
)
from oauth2_authcodeflow.exceptions import (
    OIDCConfigurationError,
    OIDCTokenError,
)
from oauth2_authcodeflow.jwt_validation import (
    validate_access_token,
    validate_id_token,
    validate_logout_token,
)
from oauth2_authcodeflow.locks import (
    active_keys,
    keyed_lock,
)
from oauth2_authcodeflow.provider import get_provider
from oauth2_authcodeflow.testing import make_signed_token


# --- settings -------------------------------------------------------------------------
def test_defaults_are_secure():
    s = Settings()
    assert s.OIDC_RP_USE_PKCE is True
    assert s.OIDC_UNUSABLE_PASSWORD is False
    assert s.OIDC_REVOKE_ON_LOGOUT is False
    assert 'HS256' not in s.OIDC_RP_SIGN_ALGOS_ALLOWED
    assert 'offline_access' not in s.OIDC_RP_SCOPES
    assert s.OIDC_SESSION_MAX_AGE_SECONDS is None


@pytest.mark.parametrize('old,new,value,expected', [
    ('OIDC_TIMEOUT', 'OIDC_HTTP_TIMEOUT', 4, 4),
    ('OIDC_RP_FORCE_SECRET_WITH_PKCE', 'OIDC_RP_CLIENT_AUTH_METHOD', True, 'client_secret_post'),
    ('OIDC_MIDDLEWARE_NO_AUTH_URL_PATTERNS', 'OIDC_LOGIN_REQUIRED_EXEMPT_URL_PATTERNS', ['^/x'], ['^/x']),
    ('OIDC_MIDDLEWARE_NO_REFRESH_URL_PATTERNS', 'OIDC_SESSION_EXEMPT_URL_PATTERNS', ['^/y'], ['^/y']),
    ('OIDC_MIDDLEWARE_API_URL_PATTERNS', 'OIDC_API_URL_PATTERNS', ['^/z'], ['^/z']),
    ('OIDC_MIDDLEWARE_LOGIN_REQUIRED_REDIRECT', 'OIDC_NAVIGATION_REDIRECT', False, False),
])
def test_deprecated_names_are_read_with_a_warning(old, new, value, expected):
    from django.conf import settings as django_settings
    with override_settings(**{old: value}):
        if hasattr(django_settings, new):
            delattr(django_settings, new)  # only the deprecated name is configured
        s = Settings()
        with pytest.warns(DeprecationWarning, match=old):
            assert getattr(s, new) == expected


def test_new_name_wins_over_deprecated_name():
    with override_settings(OIDC_TIMEOUT=4, OIDC_HTTP_TIMEOUT=9):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert Settings().OIDC_HTTP_TIMEOUT == 9


def test_removed_settings_warn_once():
    with override_settings(OIDC_MIDDLEWARE_SESSION_TIMEOUT_SECONDS=10):
        s = Settings()
        with pytest.warns(DeprecationWarning, match='no effect'):
            s.warn_removed()


def test_wrong_type_is_a_configuration_error():
    with override_settings(OIDC_REFRESH_SKEW_SECONDS='60'):
        with pytest.raises(ImproperlyConfigured):
            Settings().OIDC_REFRESH_SKEW_SECONDS


# --- encryption -----------------------------------------------------------------------
def test_derived_keys_round_trip_and_survive_secret_rotation():
    with override_settings(SECRET_KEY='old-secret'):
        token = get_cipher().encrypt('the-token')
        assert get_cipher().derived is True
    with override_settings(SECRET_KEY='new-secret', SECRET_KEY_FALLBACKS=['old-secret']):
        cipher = get_cipher()
        assert cipher.decrypt(token) == 'the-token'
        rotated = cipher.rotate(token)
    with override_settings(SECRET_KEY='new-secret', SECRET_KEY_FALLBACKS=[]):
        assert get_cipher().decrypt(rotated) == 'the-token'
        with pytest.raises(TokenDecryptError):
            get_cipher().decrypt(token)


def test_explicit_keys_take_priority_and_rotate():
    a, b = Fernet.generate_key().decode(), Fernet.generate_key().decode()
    with override_settings(OIDC_TOKEN_ENCRYPTION_KEYS=[a]):
        token = get_cipher().encrypt('x')
        assert get_cipher().derived is False
    with override_settings(OIDC_TOKEN_ENCRYPTION_KEYS=[b, a]):
        assert get_cipher().decrypt(token) == 'x'


def test_an_explicit_key_still_decrypts_what_the_derived_key_wrote():
    """Moving from the default (a key derived from SECRET_KEY) to OIDC_TOKEN_ENCRYPTION_KEYS:
    tokens written before stay readable, a rotation moves them onto the explicit key, and from
    then on they no longer depend on SECRET_KEY."""
    key = Fernet.generate_key().decode()
    with override_settings(SECRET_KEY='the-secret'):
        written_before = get_cipher().encrypt('the-token')
    with override_settings(SECRET_KEY='the-secret', OIDC_TOKEN_ENCRYPTION_KEYS=[key]):
        cipher = get_cipher()
        assert cipher.derived is False
        assert cipher.decrypt(written_before) == 'the-token'
        rotated = cipher.rotate(written_before)
    assert Fernet(key.encode()).decrypt(rotated.encode()) == b'the-token'
    with override_settings(SECRET_KEY='another-secret', SECRET_KEY_FALLBACKS=[], OIDC_TOKEN_ENCRYPTION_KEYS=[key]):
        assert get_cipher().decrypt(rotated) == 'the-token'


def test_invalid_explicit_key_is_rejected():
    with pytest.raises(ImproperlyConfigured):
        with override_settings(OIDC_TOKEN_ENCRYPTION_KEYS=['not-a-fernet-key']):
            TokenCipher.from_settings()


# --- circuit breaker ------------------------------------------------------------------
def test_breaker_opens_after_failures_and_lets_one_probe_through():
    b = CircuitBreaker('t', failures=3, window=30, cooldown=0.05, config_cooldown=10)
    for _ in range(2):
        b.failure('transient')
    assert b.state == CLOSED and b.try_acquire()
    b.failure('transient')
    assert b.is_open() and not b.try_acquire()
    time.sleep(0.06)
    assert b.state == HALF_OPEN
    assert b.try_acquire() is True
    assert b.try_acquire() is False  # only one probe
    b.failure('transient')
    assert b.state == OPEN
    time.sleep(0.06)
    assert b.try_acquire()
    b.success()
    assert b.state == CLOSED and not b.is_open()


def test_breaker_opens_immediately_on_configuration_errors():
    b = CircuitBreaker('t', failures=5, window=30, cooldown=1, config_cooldown=60)
    b.failure('config')
    assert b.is_open()
    assert 55 <= b.retry_after() <= 60


# --- keyed locks ----------------------------------------------------------------------
def test_keyed_lock_excludes_and_cleans_up():
    first = keyed_lock('k')
    assert first.acquire()
    second = keyed_lock('k')
    assert second.acquire(blocking=False) is False
    assert second.acquire(timeout=0.01) is False
    second.release()
    first.release()
    assert active_keys() == 0


def test_keyed_lock_serialises_threads():
    counter = {'value': 0, 'max': 0}

    def work():
        with keyed_lock('serial'):
            counter['value'] += 1
            counter['max'] = max(counter['max'], counter['value'])
            time.sleep(0.005)
            counter['value'] -= 1

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert counter['max'] == 1 and active_keys() == 0


# --- token-endpoint classification ----------------------------------------------------
@pytest.mark.parametrize('status,body,expected', [
    (200, {'access_token': 'a', 'expires_in': 300}, token_endpoint.OK),
    (400, {'error': 'invalid_grant', 'error_description': 'Session not active'}, token_endpoint.TERMINAL),
    (400, {'error': 'invalid_grant', 'error_description': 'Token is not active'}, token_endpoint.TERMINAL),
    (400, {'error': 'invalid_grant', 'error_description': "Invalid token issuer. Expected 'http://x'"}, token_endpoint.CONFIG),
    (401, {'error': 'invalid_client'}, token_endpoint.CONFIG),
    (400, {'error': 'unauthorized_client'}, token_endpoint.CONFIG),
    (400, {'error': 'unsupported_grant_type'}, token_endpoint.CONFIG),
    (400, {'error': 'invalid_scope'}, token_endpoint.CONFIG),
    (400, {'error': 'invalid_request'}, token_endpoint.CONFIG),
    (503, {'error': 'temporarily_unavailable'}, token_endpoint.TRANSIENT),
    (429, None, token_endpoint.TRANSIENT),            # rate limit page
    (403, None, token_endpoint.TRANSIENT),            # WAF / proxy HTML page
    (502, None, token_endpoint.TRANSIENT),
    (200, {}, token_endpoint.TRANSIENT),              # 2xx without an access token
    (200, None, token_endpoint.TRANSIENT),            # HTML 200
])
def test_classification(status, body, expected):
    assert token_endpoint.classify(status, body, 'text/html' if body is None else 'application/json')[0] == expected


@pytest.mark.parametrize('error', [requests.ConnectionError('down'), requests.Timeout('slow'), requests.exceptions.SSLError('tls')])
def test_network_errors_are_transient(oidc_op, error):
    oidc_op.script_token(error)
    outcome = token_endpoint.refresh('rt')
    assert outcome.category == token_endpoint.TRANSIENT and outcome.description == type(error).__name__


@pytest.mark.parametrize('method,secret_in_body,basic', [
    (None, True, False), ('client_secret_post', True, False), ('client_secret_basic', False, True), ('none', False, False),
])
def test_client_authentication_methods(oidc_op, method, secret_in_body, basic):
    with override_settings(OIDC_RP_CLIENT_AUTH_METHOD=method):
        token_endpoint.refresh(oidc_op.issue_tokens()['refresh_token'])
    call, auth = oidc_op.token_calls[-1], oidc_op.token_auth[-1]
    assert call['client_id'] == 'test-client'
    assert ('client_secret' in call) is secret_in_body
    assert (auth == ('test-client', 'test-secret')) is basic


def test_revoke_and_userinfo(oidc_op):
    assert token_endpoint.revoke('rt-1') is True
    assert oidc_op.revoke_calls[0]['token'] == 'rt-1'
    info = token_endpoint.fetch_userinfo(oidc_op.issue_tokens('u9')['access_token'])
    assert info['sub'] == 'u9'


# --- provider -------------------------------------------------------------------------
def test_discovery_is_cached(oidc_op):
    get_provider().metadata()
    get_provider().metadata()
    assert oidc_op.discovery_calls == 1


def test_stale_discovery_document_is_kept_when_the_op_fails(oidc_op):
    with override_settings(OIDC_DISCOVERY_CACHE_SECONDS=0):
        first = get_provider().metadata()
        oidc_op.discovery_status = 503
        assert get_provider().metadata() is first


def test_issuer_mismatch_is_a_configuration_error(oidc_op):
    with override_settings(OIDC_OP_ISSUER='https://other.example.test/realms/test'):
        with pytest.raises(OIDCConfigurationError):
            get_provider().metadata()


def test_endpoint_overrides_apply_on_top_of_discovery(oidc_op):
    with override_settings(OIDC_OP_TOKEN_URL='https://internal.example/token'):
        md = get_provider().metadata()
    assert md.token_endpoint == 'https://internal.example/token'
    assert md.authorization_endpoint == oidc_op.authorization_endpoint


def test_unknown_kid_refetches_jwks_at_most_once_per_interval(oidc_op):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = make_signed_token({'iss': oidc_op.issuer, 'sub': 's', 'aud': 'test-client', 'iat': int(time.time()),
                               'exp': int(time.time()) + 60}, key=other, kid='rotated')
    for _ in range(3):
        with pytest.raises(OIDCTokenError, match='unknown_signing_key'):
            validate_id_token(token)
    assert oidc_op.jwks_calls == 2  # initial fetch + one refetch


def test_static_signing_key_overrides_jwks(oidc_op):
    from oauth2_authcodeflow.testing import test_private_key
    pem = test_private_key().public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    with override_settings(OIDC_RP_IDP_SIGN_KEY=pem):
        claims = validate_id_token(oidc_op.issue_tokens('s1')['id_token'])
    assert claims['sub'] == 's1' and oidc_op.jwks_calls == 0


# --- JWT validation -------------------------------------------------------------------
def _id_claims(op, **extra):
    now = int(time.time())
    claims = {'iss': op.issuer, 'sub': 'u1', 'aud': 'test-client', 'azp': 'test-client', 'iat': now, 'exp': now + 300, 'typ': 'ID'}
    claims.update(extra)
    return claims


def test_valid_id_token(oidc_op):
    assert validate_id_token(oidc_op.sign(_id_claims(oidc_op, nonce='n1')), nonce='n1')['sub'] == 'u1'


@pytest.mark.parametrize('extra,reason', [
    ({'aud': 'other-client'}, 'invalid_audience'),
    ({'iss': 'https://evil.example/realms/test'}, 'invalid_issuer'),
    ({'exp': int(time.time()) - 3600}, 'token_expired'),
    ({'typ': 'Bearer'}, 'invalid_token_type'),
    ({'azp': 'other-client'}, 'invalid_azp'),
    ({'nonce': 'other'}, 'invalid_nonce'),
])
def test_invalid_id_tokens(oidc_op, extra, reason):
    with pytest.raises(OIDCTokenError, match=reason):
        validate_id_token(oidc_op.sign(_id_claims(oidc_op, **extra)), nonce='n1' if 'nonce' in extra else None)


def test_symmetric_and_none_algorithms_are_never_accepted(oidc_op):
    claims = _id_claims(oidc_op)
    hs256 = jwt.encode(claims, 'test-secret', algorithm='HS256', headers={'kid': oidc_op.kid})
    header = urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').rstrip(b'=').decode()
    payload = urlsafe_b64encode(jwt.api_jws.json.dumps(claims).encode()).rstrip(b'=').decode()
    unsigned = f'{header}.{payload}.'
    with override_settings(OIDC_RP_SIGN_ALGOS_ALLOWED=['HS256', 'none', 'RS256']):
        for token in (hs256, unsigned):
            with pytest.raises(OIDCTokenError, match='algorithm_not_allowed'):
                validate_id_token(token)
            with pytest.raises(OIDCTokenError, match='algorithm_not_allowed'):
                validate_access_token(token)


def test_access_token_validation(oidc_op):
    tokens = oidc_op.issue_tokens('u2')
    assert validate_access_token(tokens['access_token'])['sub'] == 'u2'
    with pytest.raises(OIDCTokenError, match='invalid_token_type'):
        validate_access_token(tokens['id_token'])  # an id token is not a bearer credential
    foreign = oidc_op.sign(dict(jwt.decode(tokens['access_token'], options={'verify_signature': False}), aud=['account'], azp='test-client'))
    with pytest.raises(OIDCTokenError, match='invalid_audience'):
        validate_access_token(foreign)
    with override_settings(OIDC_BEARER_ACCEPT_AZP=True):
        assert validate_access_token(foreign)['sub'] == 'u2'


@pytest.mark.parametrize('kwargs,reason', [
    ({'events': False, 'sid': 's'}, 'missing_logout_event'),
    ({'sid': 's', 'extra': {'nonce': 'x'}}, 'nonce_not_allowed'),
    ({}, 'missing_sid_and_sub'),
    ({'sid': 's', 'iat': int(time.time()) - 3600}, 'stale_logout_token'),
    ({'sid': 's', 'iat': int(time.time()) + 3600}, 'token_not_yet_valid|stale_logout_token'),
    ({'sid': 's', 'extra': {'aud': 'other'}}, 'invalid_audience'),
])
def test_invalid_logout_tokens(oidc_op, kwargs, reason):
    with pytest.raises(OIDCTokenError, match=reason):
        validate_logout_token(oidc_op.logout_token(**kwargs))


def test_valid_logout_token(oidc_op):
    assert validate_logout_token(oidc_op.logout_token(sid='s1'))['sid'] == 's1'
    assert settings.OIDC_BACKCHANNEL_LOGOUT_ENABLED
