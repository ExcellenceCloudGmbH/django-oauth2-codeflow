"""
2.1 bridge: what clients receive when authentication is needed, and what happens to the session.

Intent (plan §3.4 and §6.1):
- page navigations are redirected to the login; fetch/XHR and API requests get a JSON 401 with a machine readable code,
  `WWW-Authenticate` and `Cache-Control: no-store` — including GETs to API URLs that 2.0 redirected;
- an anonymous session is never destroyed (a pending login's state and nonce survive concurrent XHRs);
- a 3.0 session is treated as unauthenticated without destroying it or blacklisting anything;
- no token ever appears in a response body or a log record.
"""
import re
from datetime import (
    datetime,
    timezone,
)
from json import loads
from logging import DEBUG
from unittest.mock import (
    MagicMock,
    patch,
)

import jwt
import pytest
import requests
from django.contrib.auth import BACKEND_SESSION_KEY
from django.contrib.sessions.middleware import SessionMiddleware
from django.core.cache import caches
from django.http import HttpResponse

from oauth2_authcodeflow import constants
from oauth2_authcodeflow.middleware import (
    BearerAuthMiddleware,
    LoginRequiredMiddleware,
    RefreshAccessTokenMiddleware,
    RefreshSessionMiddleware,
)
from oauth2_authcodeflow.models import BlacklistedToken

pytestmark = pytest.mark.django_db

OIDC_BACKEND = 'oauth2_authcodeflow.auth.AuthenticationBackend'
JWT_PATTERN = re.compile(r'eyJ[\w-]+\.[\w-]+\.[\w-]*')


def now_ts():
    return int(datetime.now(timezone.utc).timestamp())


def make_jwt(**claims):
    claims.setdefault('exp', now_ts() + 3600)
    claims.setdefault('email', 'user@example.com')
    return jwt.encode(claims, 'k' * 32, algorithm='HS256')


def view(request):
    return HttpResponse('view')


@pytest.fixture
def oidc_settings(settings):
    settings.OIDC_RP_CLIENT_ID = 'a_client'
    settings.OIDC_RP_CLIENT_SECRET = 'secret'
    settings.OIDC_RP_USE_PKCE = False
    settings.OIDC_REFRESH_LOCK_CACHE_ALIAS = 'default'
    caches['default'].clear()
    return settings


def anonymous_request(rf, path='/page', method='get', session_data=None, **headers):
    request = getattr(rf, method)(path, **headers)
    SessionMiddleware(lambda r: HttpResponse()).process_request(request)
    for key, value in (session_data or {}).items():
        request.session[key] = value
    if session_data:
        request.session.save()
        session_key = request.session.session_key
        request.session = request.session.__class__(session_key=session_key)
    request.user = MagicMock(is_authenticated=False)
    return request


def oidc_request(rf, *, path='/page', expires_in=-20, session_data=None, **headers):
    data = {
        BACKEND_SESSION_KEY: OIDC_BACKEND,
        constants.SESSION_ID_TOKEN: make_jwt(jti='id'),
        constants.SESSION_ACCESS_TOKEN: make_jwt(jti='at'),
        constants.SESSION_REFRESH_TOKEN: make_jwt(jti='rt'),
        constants.SESSION_ACCESS_EXPIRES_AT: now_ts() + expires_in,
        constants.SESSION_EXPIRES_AT: now_ts() + 3600,
        constants.SESSION_OP_TOKEN_URL: 'https://op.example/token',
    }
    data.update(session_data or {})
    request = anonymous_request(rf, path=path, session_data=data, **headers)
    request.user = MagicMock(is_authenticated=True)
    return request


class TestRequestKinds:
    @pytest.mark.parametrize('method, path, headers, expected', [
        ('get', '/page', {}, 302),  # no signal at all: the historical navigation behaviour
        ('get', '/page', {'HTTP_SEC_FETCH_MODE': 'navigate', 'HTTP_ACCEPT': 'text/html'}, 302),
        ('get', '/page', {'HTTP_SEC_FETCH_MODE': 'navigate', 'HTTP_SEC_FETCH_DEST': 'iframe'}, 302),
        ('get', '/page', {'HTTP_SEC_FETCH_DEST': 'document'}, 302),
        ('get', '/page', {'HTTP_SEC_FETCH_MODE': 'cors'}, 401),
        ('get', '/page', {'HTTP_SEC_FETCH_MODE': 'same-origin'}, 401),
        ('get', '/page', {'HTTP_SEC_FETCH_DEST': 'empty'}, 401),
        ('get', '/page', {'HTTP_X_REQUESTED_WITH': 'XMLHttpRequest'}, 401),
        ('get', '/page', {'HTTP_ACCEPT': 'application/json'}, 401),
        ('get', '/page', {'HTTP_AUTHORIZATION': 'Bearer abc'}, 401),
        ('get', '/api/things', {'HTTP_SEC_FETCH_MODE': 'cors'}, 401),  # 2.0 redirected every GET, even fetches
        ('get', '/api/things', {'HTTP_SEC_FETCH_MODE': 'navigate'}, 302),  # a page navigation to an API URL
        ('post', '/api/things', {}, 401),
        ('post', '/page', {}, 302),  # a classic form post
        ('post', '/page', {'HTTP_SEC_FETCH_MODE': 'cors'}, 401),
    ])
    def test_login_required(self, method, path, headers, expected, oidc_settings, rf):
        request = anonymous_request(rf, path=path, method=method, **headers)
        response = LoginRequiredMiddleware(view)(request)
        assert response.status_code == expected
        assert response['Cache-Control'] == 'no-store'
        if expected == 401:
            assert loads(response.content) == {'error': 'not_authenticated', 'login_url': '/oidc/authenticate', 'token_type': 'id_token'}
            scheme = 'Bearer' if 'HTTP_AUTHORIZATION' in headers else 'Session'
            assert response['WWW-Authenticate'] == f'{scheme} realm="a_client"'
        else:
            assert response.url.startswith('/oidc/authenticate?')

    def test_login_required_without_redirect(self, oidc_settings, rf):
        oidc_settings.OIDC_MIDDLEWARE_LOGIN_REQUIRED_REDIRECT = False
        response = LoginRequiredMiddleware(view)(anonymous_request(rf, path='/page', HTTP_SEC_FETCH_MODE='navigate'))
        assert response.status_code == 401

    @patch('oauth2_authcodeflow.middleware.request_post')
    def test_ended_session(self, request_post, oidc_settings, rf):
        request_post.return_value = MagicMock(status_code=400, **{'json.return_value': {'error': 'invalid_grant'}})
        response = RefreshAccessTokenMiddleware(view)(oidc_request(rf, HTTP_SEC_FETCH_MODE='cors'))
        assert response.status_code == 401
        assert loads(response.content)['error'] == 'session_expired'
        request_post.return_value = MagicMock(status_code=400, **{'json.return_value': {'error': 'invalid_grant'}})
        response = RefreshAccessTokenMiddleware(view)(oidc_request(rf, HTTP_SEC_FETCH_MODE='navigate', HTTP_ACCEPT='text/html'))
        assert response.status_code == 302
        assert response.url.startswith('/oidc/authenticate?')


class TestNoAnonymousFlush:
    @pytest.mark.parametrize('headers', [{'HTTP_SEC_FETCH_MODE': 'cors'}, {'HTTP_SEC_FETCH_MODE': 'navigate'}, {}])
    def test_pending_login_survives_concurrent_requests(self, headers, oidc_settings, rf):
        pending = {constants.SESSION_STATE: 'the-state', constants.SESSION_NONCE: 'the-nonce', constants.SESSION_NEXT_URL: '/back'}
        request = anonymous_request(rf, path='/api/things', session_data=pending, **headers)
        session_key = request.session.session_key
        response = LoginRequiredMiddleware(view)(request)
        assert response.status_code in (302, 401)
        assert request.session.session_key == session_key
        stored = request.session.__class__(session_key=session_key)
        assert stored[constants.SESSION_STATE] == 'the-state'
        assert stored[constants.SESSION_NONCE] == 'the-nonce'


class TestAuthUnavailable:
    @patch('oauth2_authcodeflow.middleware.request_post')
    def test_api_and_navigation(self, request_post, oidc_settings, rf):
        request_post.side_effect = requests.ConnectionError('down')
        api = RefreshAccessTokenMiddleware(view)(oidc_request(rf, HTTP_SEC_FETCH_MODE='cors'))
        assert api.status_code == 503
        assert loads(api.content) == {'error': 'auth_unavailable', 'retry_after': 5}
        assert api['Retry-After'] == '5'
        assert api['Cache-Control'] == 'no-store'
        page = RefreshAccessTokenMiddleware(view)(oidc_request(rf, path='/page?x=<b>', HTTP_SEC_FETCH_MODE='navigate'))
        assert page.status_code == 503
        assert page['Content-Type'].startswith('text/html')
        assert b'<meta http-equiv="refresh" content="5">' in page.content
        assert b'<b>' not in page.content  # the retry link is escaped
        oidc_settings.OIDC_RETRY_AFTER_SECONDS = 12
        api = RefreshAccessTokenMiddleware(view)(oidc_request(rf, HTTP_SEC_FETCH_MODE='cors'))
        assert api['Retry-After'] == '12'


class TestOidc3Session:
    def oidc3_request(self, rf, **headers):
        data = {BACKEND_SESSION_KEY: OIDC_BACKEND, '_auth_user_id': '1', constants.SESSION_OIDC3_ID: 'b9a1c3f0-uuid'}
        request = anonymous_request(rf, path='/api/things', session_data=data, **headers)
        request.user = MagicMock(is_authenticated=True)
        return request

    @pytest.mark.parametrize('middleware_class', [RefreshSessionMiddleware, RefreshAccessTokenMiddleware])
    def test_treated_as_unauthenticated_without_destroying_it(self, middleware_class, oidc_settings, rf):
        request = self.oidc3_request(rf, HTTP_SEC_FETCH_MODE='cors')
        session_key = request.session.session_key
        response = middleware_class(view)(request)
        assert response.status_code == 401
        assert loads(response.content)['error'] == 'session_expired'
        assert request.session.session_key == session_key
        assert request.session.__class__(session_key=session_key)[constants.SESSION_OIDC3_ID] == 'b9a1c3f0-uuid'
        assert BlacklistedToken.objects.count() == 0
        navigation = middleware_class(view)(self.oidc3_request(rf, HTTP_SEC_FETCH_MODE='navigate'))
        assert navigation.status_code == 302

    def test_login_required_lets_it_through_to_the_refresh_checks(self, oidc_settings, rf):
        assert LoginRequiredMiddleware(view)(self.oidc3_request(rf)).content == b'view'

    def test_a_2_1_login_on_top_of_it_works(self, oidc_settings, rf):
        request = oidc_request(rf, expires_in=300, session_data={constants.SESSION_OIDC3_ID: 'b9a1c3f0-uuid'})
        stack = RefreshSessionMiddleware(RefreshAccessTokenMiddleware(view))
        assert stack(request).content == b'view'


class TestTokenHygiene:
    @patch('oauth2_authcodeflow.middleware.request_post')
    def test_no_token_in_bodies_or_logs(self, request_post, oidc_settings, rf, caplog):
        caplog.set_level(DEBUG)
        bodies = []
        leaked = make_jwt(jti='leaked')
        # the OP echoes a token in its error description: it must not be relayed
        request_post.return_value = MagicMock(status_code=500, **{'json.return_value': {'error': 'server_error', 'error_description': f'bad {leaked}'}})
        request_post.return_value.text = f'bad {leaked}'
        for headers in ({'HTTP_SEC_FETCH_MODE': 'cors'}, {'HTTP_SEC_FETCH_MODE': 'navigate'}):
            bodies.append(RefreshAccessTokenMiddleware(view)(oidc_request(rf, **headers)).content)
        request_post.return_value = MagicMock(status_code=400, **{'json.return_value': {'error': 'invalid_grant', 'error_description': f'bad {leaked}'}})
        bodies.append(RefreshAccessTokenMiddleware(view)(oidc_request(rf, HTTP_SEC_FETCH_MODE='cors')).content)
        expired = oidc_request(rf, HTTP_SEC_FETCH_MODE='cors', session_data={constants.SESSION_EXPIRES_AT: now_ts() - 1})
        bodies.append(RefreshSessionMiddleware(view)(expired).content)
        bodies.append(LoginRequiredMiddleware(view)(anonymous_request(rf, HTTP_SEC_FETCH_MODE='cors')).content)
        # a blacklisted bearer token
        bearer = make_jwt(jti='bearer', aud='a_client')
        BlacklistedToken.blacklist(bearer)
        with patch('oauth2_authcodeflow.auth.BearerAuthenticationBackend.get_oidc_urls', return_value={}):
            request = anonymous_request(rf, HTTP_AUTHORIZATION=f'Bearer {bearer}')
            assert BearerAuthMiddleware(view)(request).content == b'view'
        for body in bodies:
            assert not JWT_PATTERN.search(body.decode('utf8', 'replace')), body
        assert not JWT_PATTERN.search(caplog.text), caplog.text
        assert 'event=oidc.session.ended reason=refresh_invalid_grant' in caplog.text
        assert 'event=oidc.session.ended reason=session_idle_timeout' in caplog.text
