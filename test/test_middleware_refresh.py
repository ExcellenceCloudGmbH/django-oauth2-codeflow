"""
2.1 bridge: the token refresh must never turn an OP hiccup or a concurrent request into a logout.

Intent (plan §6.1 "2.1.0 — bridge release"):
- only an ``invalid_grant`` answer ends a session; every other failure keeps it and serves the still valid
  access token, or answers 503 with Retry-After;
- one refresh at a time per session, released only by its owner, and a broken lock cache never causes a stampede;
- concurrent requests never write the session while waiting for the leader;
- a successful refresh moves the Django idle timeout forward;
- a stale session snapshot holding an old (even blacklisted) id token is not logged out.
"""
import threading
import time
from datetime import (
    datetime,
    timezone,
)
from json import loads
from unittest.mock import (
    MagicMock,
    patch,
)

import jwt
import pytest
import requests
from django.contrib.auth import BACKEND_SESSION_KEY
from django.contrib.sessions.backends.cache import SessionStore as CacheSessionStore
from django.contrib.sessions.middleware import SessionMiddleware
from django.core.cache import caches
from django.http import HttpResponse

from oauth2_authcodeflow import constants
from oauth2_authcodeflow.middleware import (
    LoginRequiredMiddleware,
    RefreshAccessTokenMiddleware,
    RefreshLock,
    RefreshSessionMiddleware,
    classify_token_response,
)
from oauth2_authcodeflow.models import BlacklistedToken

pytestmark = pytest.mark.django_db

OIDC_BACKEND = 'oauth2_authcodeflow.auth.AuthenticationBackend'
SIGNING_KEY = 'k' * 32


def make_jwt(**claims):
    claims.setdefault('exp', int(datetime.now(timezone.utc).timestamp()) + 3600)
    claims.setdefault('email', 'user@example.com')
    return jwt.encode(claims, SIGNING_KEY, algorithm='HS256')


OLD_ID_TOKEN = make_jwt(sid='s1', jti='id-old')
ACCESS_TOKEN = make_jwt(jti='at-old')
REFRESH_TOKEN = make_jwt(jti='rt-old')
NEW_ACCESS_TOKEN = make_jwt(jti='at-new')
NEW_ID_TOKEN = make_jwt(sid='s1', jti='id-new')


def now_ts():
    return int(datetime.now(timezone.utc).timestamp())


def fake_response(status, body):
    resp = MagicMock(status_code=status)
    if isinstance(body, (dict, list)):
        resp.json.return_value = body
    else:
        resp.json.side_effect = requests.JSONDecodeError("Expecting value", str(body), 0)
    resp.text = str(body)
    return resp


@pytest.fixture
def oidc_settings(settings):
    settings.OIDC_RP_CLIENT_ID = 'client_id'
    settings.OIDC_RP_CLIENT_SECRET = 'client_secret'
    settings.OIDC_RP_USE_PKCE = False
    settings.OIDC_REFRESH_SKEW_SECONDS = 30
    settings.OIDC_RETRY_AFTER_SECONDS = 5
    settings.OIDC_REFRESH_LOCK_CACHE_ALIAS = 'default'
    settings.OIDC_MIDDLEWARE_SESSION_TIMEOUT_SECONDS = 3600
    caches['default'].clear()
    return settings


def build_session(sf, rf, *, expires_in, extra=None):
    """Store an OIDC session (2.x layout) and return a fresh request bound to it, as a real request would be."""
    seed = rf.get('/')
    session = sf(seed)
    session[BACKEND_SESSION_KEY] = OIDC_BACKEND
    session[constants.SESSION_ID_TOKEN] = OLD_ID_TOKEN
    session[constants.SESSION_ACCESS_TOKEN] = ACCESS_TOKEN
    session[constants.SESSION_REFRESH_TOKEN] = REFRESH_TOKEN
    session[constants.SESSION_ACCESS_EXPIRES_AT] = now_ts() + expires_in
    session[constants.SESSION_EXPIRES_AT] = now_ts() + 3600
    session[constants.SESSION_OP_TOKEN_URL] = 'https://op.example/token'
    for key, value in (extra or {}).items():
        session[key] = value
    session.save()
    return session.session_key


def bound_request(rf, session_key, path='/api/things', **headers):
    request = rf.get(path, **headers)
    SessionMiddleware(lambda r: HttpResponse()).process_request(request)
    request.session = request.session.__class__(session_key=session_key)
    request.user = MagicMock(is_authenticated=True)
    return request


def view(request):
    return HttpResponse('view')


class TestClassification:
    @pytest.mark.parametrize('status, body, expected', [
        (200, {'access_token': 'x', 'expires_in': 300}, ('ok', None)),
        (400, {'error': 'invalid_grant', 'error_description': 'Token is not active'}, ('terminal', 'invalid_grant')),
        (400, {'error': 'invalid_grant', 'error_description': 'Session not active'}, ('terminal', 'invalid_grant')),
        (400, {'error': 'invalid_grant', 'error_description': 'Invalid token issuer. Expected https://a'}, ('config', 'invalid_grant')),
        (401, {'error': 'invalid_client'}, ('config', 'invalid_client')),
        (400, {'error': 'unauthorized_client'}, ('config', 'unauthorized_client')),
        (503, {'error': 'temporarily_unavailable'}, ('config', 'temporarily_unavailable')),
        (429, '<html>Too Many Requests</html>', ('transient', None)),
        (500, '<html>Internal Server Error</html>', ('transient', None)),
        (502, '<html>Bad gateway</html>', ('transient', None)),
        (403, '<html>Attention Required! | Cloudflare</html>', ('transient', None)),
        (200, {'token_type': 'Bearer'}, ('transient', None)),
        (200, 'not json', ('transient', None)),
        (200, ['a', 'list'], ('transient', None)),
    ])
    def test_classify_token_response(self, status, body, expected):
        assert classify_token_response(status, body if isinstance(body, (dict, list)) else None) == expected


class TestRefreshOutcomes:
    TERMINAL = [
        (400, {'error': 'invalid_grant', 'error_description': 'Token is not active'}),
        (400, {'error': 'invalid_grant', 'error_description': 'Session not active'}),
    ]
    KEPT = [
        (400, {'error': 'invalid_grant', 'error_description': 'Invalid token issuer'}),
        (401, {'error': 'invalid_client'}),
        (400, {'error': 'unauthorized_client'}),
        (503, {'error': 'temporarily_unavailable'}),
        (429, '<html>Too Many Requests</html>'),
        (500, '<html>oops</html>'),
        (502, '<html>Bad gateway</html>'),
        (504, '<html>Gateway timeout</html>'),
        (403, '<html>Attention Required! | Cloudflare</html>'),
        (200, {'token_type': 'Bearer'}),
        (200, 'not json'),
    ]

    @pytest.mark.parametrize('status, body', TERMINAL)
    @patch('oauth2_authcodeflow.middleware.request_post')
    def test_invalid_grant_ends_the_session(self, request_post, status, body, oidc_settings, rf, sf, caplog):
        request_post.return_value = fake_response(status, body)
        session_key = build_session(sf, rf, expires_in=-20)
        request = bound_request(rf, session_key, HTTP_SEC_FETCH_MODE='cors')
        response = RefreshAccessTokenMiddleware(view)(request)
        assert response.status_code == 401
        assert loads(response.content)['error'] == 'session_expired'
        assert request.session.session_key is None  # flushed: the user must sign in again
        assert not request.session.__class__(session_key=session_key).exists(session_key)
        assert 'event=oidc.session.ended reason=refresh_invalid_grant' in caplog.text

    @pytest.mark.parametrize('status, body', KEPT)
    @patch('oauth2_authcodeflow.middleware.request_post')
    def test_non_definitive_failure_keeps_the_session_and_answers_503_when_expired(self, request_post, status, body, oidc_settings, rf, sf):
        request_post.return_value = fake_response(status, body)
        session_key = build_session(sf, rf, expires_in=-20)
        request = bound_request(rf, session_key, HTTP_SEC_FETCH_MODE='cors')
        response = RefreshAccessTokenMiddleware(view)(request)
        assert response.status_code == 503
        assert loads(response.content) == {'error': 'auth_unavailable', 'retry_after': 5}
        assert response['Retry-After'] == '5'
        assert response['Cache-Control'] == 'no-store'
        stored = request.session.__class__(session_key=session_key)
        assert stored[constants.SESSION_REFRESH_TOKEN] == REFRESH_TOKEN  # the session survives
        assert request.session.session_key == session_key

    @pytest.mark.parametrize('status, body', KEPT)
    @patch('oauth2_authcodeflow.middleware.request_post')
    def test_non_definitive_failure_serves_the_still_valid_token(self, request_post, status, body, oidc_settings, rf, sf):
        request_post.return_value = fake_response(status, body)
        session_key = build_session(sf, rf, expires_in=10)  # inside the refresh skew, still valid
        request = bound_request(rf, session_key)
        response = RefreshAccessTokenMiddleware(view)(request)
        assert response.content == b'view'
        assert request.session[constants.SESSION_ACCESS_TOKEN] == ACCESS_TOKEN
        request_post.assert_called_once()

    @pytest.mark.parametrize('exc', [requests.ConnectionError('refused'), requests.Timeout('slow'), requests.exceptions.SSLError('tls')])
    @patch('oauth2_authcodeflow.middleware.request_post')
    def test_network_errors_keep_the_session(self, request_post, exc, oidc_settings, rf, sf):
        request_post.side_effect = exc
        session_key = build_session(sf, rf, expires_in=-20)
        request = bound_request(rf, session_key, HTTP_ACCEPT='text/html', HTTP_SEC_FETCH_MODE='navigate', path='/page')
        response = RefreshAccessTokenMiddleware(view)(request)
        # a navigation gets a small self-retrying page, never a redirect to the login
        assert response.status_code == 503
        assert response['Retry-After'] == '5'
        assert b'http-equiv="refresh"' in response.content
        assert request.session.session_key == session_key

    @patch('oauth2_authcodeflow.middleware.request_post')
    def test_success_stores_tokens_and_slides_the_idle_timeout(self, request_post, oidc_settings, rf, sf):
        request_post.return_value = fake_response(200, {
            'access_token': NEW_ACCESS_TOKEN, 'expires_in': 300, 'id_token': NEW_ID_TOKEN, 'refresh_token': 'rt-new',
        })
        session_key = build_session(sf, rf, expires_in=-20)
        request = bound_request(rf, session_key)
        before = now_ts()
        response = RefreshAccessTokenMiddleware(view)(request)
        assert response.content == b'view'
        stored = request.session.__class__(session_key=session_key)
        assert stored[constants.SESSION_ACCESS_TOKEN] == NEW_ACCESS_TOKEN
        assert stored[constants.SESSION_ID_TOKEN] == NEW_ID_TOKEN
        assert stored[constants.SESSION_REFRESH_TOKEN] == 'rt-new'
        assert stored[constants.SESSION_ACCESS_EXPIRES_AT] >= before + 300
        assert stored[constants.SESSION_EXPIRES_AT] >= before + 3600

    @patch('oauth2_authcodeflow.middleware.request_post')
    def test_missing_expires_in_falls_back_to_the_token_exp(self, request_post, oidc_settings, rf, sf):
        access = make_jwt(exp=now_ts() + 200)
        request_post.return_value = fake_response(200, {'access_token': access})
        session_key = build_session(sf, rf, expires_in=-20)
        request = bound_request(rf, session_key)
        RefreshAccessTokenMiddleware(view)(request)
        assert now_ts() + 190 <= request.session[constants.SESSION_ACCESS_EXPIRES_AT] <= now_ts() + 200

    @patch('oauth2_authcodeflow.middleware.request_post')
    def test_refresh_uses_the_configured_timeout_and_proxy(self, request_post, oidc_settings, rf, sf):
        oidc_settings.OIDC_TIMEOUT = (3.05, 8)
        oidc_settings.OIDC_PROXY = {'https': 'http://proxy:3128'}
        request_post.return_value = fake_response(200, {'access_token': NEW_ACCESS_TOKEN, 'expires_in': 300})
        session_key = build_session(sf, rf, expires_in=-20)
        RefreshAccessTokenMiddleware(view)(bound_request(rf, session_key))
        assert request_post.call_args.kwargs['timeout'] == (3.05, 8)
        assert request_post.call_args.kwargs['proxies'] == {'https': 'http://proxy:3128'}


class TestRefreshLock:
    class NoneCache:
        """django_redis with IGNORE_EXCEPTIONS while Redis is down: every call answers None"""
        def add(self, *args, **kwargs):
            return None

        def get(self, *args, **kwargs):
            return None

        def delete(self, *args, **kwargs):
            return None

    class BrokenCache(NoneCache):
        def add(self, *args, **kwargs):
            raise ConnectionError("cache down")

    def test_only_the_owner_releases(self):
        cache = caches['default']
        cache.clear()
        first = RefreshLock(cache, 'oidc:refresh:k1', 20)
        assert first.acquire() is True
        second = RefreshLock(cache, 'oidc:refresh:k1', 20)
        assert second.acquire() is False
        second.release()  # not the owner: nothing happens
        assert cache.get('oidc:refresh:k1') == first.token
        # the first lock outlived its TTL and somebody else took it: a late release must not free the new owner's lock
        cache.set('oidc:refresh:k1', 'another-owner', 20)
        first.release()
        assert cache.get('oidc:refresh:k1') == 'another-owner'

    @pytest.mark.parametrize('cache_class', [NoneCache, BrokenCache])
    def test_unusable_cache_falls_back_to_a_process_lock(self, cache_class):
        first = RefreshLock(cache_class(), 'oidc:refresh:k2', 20)
        assert first.acquire() is True
        assert first.mode == 'local'
        second = RefreshLock(cache_class(), 'oidc:refresh:k2', 20)
        assert second.acquire() is False  # still single flight
        first.release()
        assert second.acquire() is True
        second.release()


class TestConcurrentRequests:
    @patch('oauth2_authcodeflow.middleware.request_post')
    def test_follower_with_a_valid_token_does_not_wait(self, request_post, oidc_settings, rf, sf):
        session_key = build_session(sf, rf, expires_in=10)
        caches['default'].add(f"oidc:refresh:{session_key}", 'leader', 20)  # another request is refreshing
        request = bound_request(rf, session_key)
        started = time.monotonic()
        response = RefreshAccessTokenMiddleware(view)(request)
        assert time.monotonic() - started < 0.5
        assert response.content == b'view'
        request_post.assert_not_called()

    @patch('oauth2_authcodeflow.middleware.request_post')
    def test_follower_waits_read_only_and_adopts_the_leaders_tokens(self, request_post, oidc_settings, rf, sf):
        session_key = build_session(sf, rf, expires_in=-20)
        caches['default'].add(f"oidc:refresh:{session_key}", 'leader', 20)
        request = bound_request(rf, session_key)
        request.session.load()
        modified_while_waiting = []

        def leader_finishes(seconds):
            modified_while_waiting.append(request.session.modified)
            peer = request.session.__class__(session_key=session_key)
            peer[constants.SESSION_ACCESS_TOKEN] = NEW_ACCESS_TOKEN
            peer[constants.SESSION_ACCESS_EXPIRES_AT] = now_ts() + 300
            peer.save()

        middleware = RefreshAccessTokenMiddleware(view)
        with patch('oauth2_authcodeflow.middleware.time.sleep', side_effect=leader_finishes):
            response = middleware(request)
        assert response.content == b'view'
        assert modified_while_waiting == [False]  # nothing written to the request's session while polling
        assert request.session[constants.SESSION_ACCESS_TOKEN] == NEW_ACCESS_TOKEN
        request_post.assert_not_called()  # never refreshes in parallel with the leader

    @patch('oauth2_authcodeflow.middleware.request_post')
    def test_follower_never_refreshes_when_the_leader_is_late(self, request_post, oidc_settings, rf, sf):
        session_key = build_session(sf, rf, expires_in=-20)
        caches['default'].add(f"oidc:refresh:{session_key}", 'leader', 20)
        request = bound_request(rf, session_key, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        middleware = RefreshAccessTokenMiddleware(view)
        middleware.REFRESH_WAIT_SECONDS = 0.05
        with patch('oauth2_authcodeflow.middleware.time.sleep'):
            response = middleware(request)
        assert response.status_code == 503
        assert response['Retry-After'] == '1'
        request_post.assert_not_called()
        assert request.session.session_key == session_key

    @patch('oauth2_authcodeflow.middleware.request_post')
    def test_follower_sees_the_session_ended_by_the_leader(self, request_post, oidc_settings, rf, sf):
        session_key = build_session(sf, rf, expires_in=-20)
        caches['default'].add(f"oidc:refresh:{session_key}", 'leader', 20)
        request = bound_request(rf, session_key, HTTP_SEC_FETCH_MODE='cors')

        def leader_got_invalid_grant(seconds):
            request.session.__class__(session_key=session_key).delete()

        with patch('oauth2_authcodeflow.middleware.time.sleep', side_effect=leader_got_invalid_grant):
            response = RefreshAccessTokenMiddleware(view)(request)
        assert response.status_code == 401
        assert loads(response.content)['error'] == 'session_expired'
        request_post.assert_not_called()

    @patch('oauth2_authcodeflow.middleware.request_post')
    def test_leader_adopts_tokens_refreshed_by_a_previous_leader(self, request_post, oidc_settings, rf, sf):
        session_key = build_session(sf, rf, expires_in=-20)
        request = bound_request(rf, session_key)
        request.session.load()  # this request read the session before the previous leader stored new tokens
        peer = request.session.__class__(session_key=session_key)
        peer[constants.SESSION_ACCESS_TOKEN] = NEW_ACCESS_TOKEN
        peer[constants.SESSION_ACCESS_EXPIRES_AT] = now_ts() + 300
        peer.save()
        response = RefreshAccessTokenMiddleware(view)(request)
        assert response.content == b'view'
        assert request.session[constants.SESSION_ACCESS_TOKEN] == NEW_ACCESS_TOKEN
        request_post.assert_not_called()

    def test_no_refresh_stampede_when_the_lock_cache_is_down(self, oidc_settings, rf):
        oidc_settings.SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
        seed = CacheSessionStore()
        seed[BACKEND_SESSION_KEY] = OIDC_BACKEND
        seed[constants.SESSION_ID_TOKEN] = OLD_ID_TOKEN
        seed[constants.SESSION_ACCESS_TOKEN] = ACCESS_TOKEN
        seed[constants.SESSION_REFRESH_TOKEN] = REFRESH_TOKEN
        seed[constants.SESSION_ACCESS_EXPIRES_AT] = now_ts() - 20
        seed[constants.SESSION_EXPIRES_AT] = now_ts() + 3600
        seed[constants.SESSION_OP_TOKEN_URL] = 'https://op.example/token'
        seed.save()
        calls = []
        calls_lock = threading.Lock()

        def slow_token_endpoint(*args, **kwargs):
            with calls_lock:
                calls.append(1)
            time.sleep(0.2)
            return fake_response(200, {'access_token': NEW_ACCESS_TOKEN, 'expires_in': 300})

        middleware = RefreshAccessTokenMiddleware(view)
        middleware.REFRESH_WAIT_STEP = 0.01
        results = []
        errors = []

        def worker():
            try:
                request = rf.get('/api/things')
                request.session = CacheSessionStore(session_key=seed.session_key)
                request.user = MagicMock(is_authenticated=True)
                response = middleware(request)
                results.append((response.content, request.session[constants.SESSION_ACCESS_TOKEN]))
            except Exception as e:  # pragma: no cover - surfaced by the assertion below
                errors.append(e)

        with patch('oauth2_authcodeflow.middleware.request_post', side_effect=slow_token_endpoint), \
                patch.object(RefreshAccessTokenMiddleware, 'get_lock_cache', return_value=TestRefreshLock.NoneCache()):
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
        assert not errors
        assert len(calls) == 1  # a single refresh despite the cache outage
        assert results == [(b'view', NEW_ACCESS_TOKEN)] * 8


class TestStaleSnapshot:
    @pytest.mark.django_db
    def test_a_blacklisted_old_id_token_in_a_stale_snapshot_does_not_log_out(self, oidc_settings, rf, sf):
        """
        2.0 blacklisted the previous id token on every refresh and checked every request against the blacklist:
        a request still holding the pre-refresh snapshot was logged out and destroyed the fresh session.
        """
        BlacklistedToken.blacklist(OLD_ID_TOKEN)
        assert BlacklistedToken.is_blacklisted(OLD_ID_TOKEN)
        session_key = build_session(sf, rf, expires_in=120)
        request = bound_request(rf, session_key)
        stack = LoginRequiredMiddleware(RefreshSessionMiddleware(RefreshAccessTokenMiddleware(view)))
        response = stack(request)
        assert response.status_code == 200
        assert response.content == b'view'
        assert request.session.session_key == session_key

    @patch('oauth2_authcodeflow.middleware.request_post')
    @pytest.mark.django_db
    def test_refresh_no_longer_blacklists_the_previous_id_token(self, request_post, oidc_settings, rf, sf):
        request_post.return_value = fake_response(200, {'access_token': NEW_ACCESS_TOKEN, 'expires_in': 300, 'id_token': NEW_ID_TOKEN})
        session_key = build_session(sf, rf, expires_in=-20)
        RefreshAccessTokenMiddleware(view)(bound_request(rf, session_key))
        assert BlacklistedToken.objects.count() == 0
