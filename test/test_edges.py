"""Edges of the contract: deactivated users, bearer refusals, undecryptable tokens, the 2.x
decorator and mixin, the HTTP client towards the OP, synchronous sockets, embedded tokens."""
import http.server
import io
import json
import threading
from datetime import timedelta
from urllib.parse import (
    parse_qs,
    urlparse,
)

import pytest
from asgiref.sync import async_to_sync
from cryptography.fernet import Fernet
from django.contrib.auth import get_user_model
from django.contrib.sessions.middleware import SessionMiddleware
from django.core.management import call_command
from django.http import HttpResponse
from django.test import (
    Client,
    RequestFactory,
    override_settings,
)
from django.urls import path
from django.utils import timezone

from oauth2_authcodeflow import sessions
from oauth2_authcodeflow.conf import get_default_django_username
from oauth2_authcodeflow.exceptions import (
    MiddlewareException,
    OIDCSessionEnded,
)
from oauth2_authcodeflow.http import (
    OPClient,
    http_timeout,
)
from oauth2_authcodeflow.jwt_validation import validate_access_token
from oauth2_authcodeflow.models import OIDCSession
from oauth2_authcodeflow.testing import create_oidc_session
from oauth2_authcodeflow.tokens import (
    get_access_token,
    get_id_claims,
)

XHR = {'HTTP_SEC_FETCH_MODE': 'cors'}


@pytest.mark.django_db
def test_deactivating_a_user_ends_the_sign_in_at_the_next_request(client, user, oidc_op):
    row = create_oidc_session(client, user)
    user.is_active = False
    user.save()
    response = client.get('/api/data/', **XHR)
    assert response.status_code == 401 and response.json()['error'] == 'session_expired'
    assert OIDCSession.objects.get(pk=row.pk).ended_at is not None


@pytest.mark.django_db
def test_bearer_refuses_inactive_and_unknown_users(client, oidc_op):
    email = 'ina@example.test'
    get_user_model().objects.create_user(get_default_django_username({'email': email}), email, is_active=False)
    inactive = oidc_op.issue_tokens('sub-ina', email=email)['access_token']
    response = client.get('/api/data/', HTTP_AUTHORIZATION=f'Bearer {inactive}')
    assert response.status_code == 401 and response['WWW-Authenticate'] == 'Bearer error="invalid_token"'
    with override_settings(OIDC_CREATE_USER=False):
        unknown = oidc_op.issue_tokens('sub-zed', email='zed@example.test')['access_token']
        assert client.get('/api/data/', HTTP_AUTHORIZATION=f'Bearer {unknown}').status_code == 401


@pytest.mark.django_db
def test_get_access_token_returns_the_bearer_token(client, oidc_op):
    token = oidc_op.issue_tokens('sub-b', email='b@example.test')['access_token']
    assert client.get('/api/fresh/', HTTP_AUTHORIZATION=f'Bearer {token}').json()['has_token'] is True


@pytest.mark.django_db
def test_undecryptable_tokens_end_the_sign_in_cleanly(client, user, oidc_op):
    old_key, new_key = Fernet.generate_key().decode(), Fernet.generate_key().decode()
    with override_settings(OIDC_TOKEN_ENCRYPTION_KEYS=[old_key]):
        row = create_oidc_session(client, user)
    with override_settings(OIDC_TOKEN_ENCRYPTION_KEYS=[new_key]):  # the old key was dropped too early
        response = client.get('/api/fresh/', **XHR)
    assert response.status_code == 401 and response.json()['error'] == 'session_expired'
    assert OIDCSession.objects.get(pk=row.pk).end_reason == 'decrypt_failed'


@pytest.mark.django_db
def test_moving_from_the_derived_key_to_an_explicit_key_keeps_every_sign_in(client, user, oidc_op):
    """The documented rotation, starting from the default key (derived from SECRET_KEY): put a key
    in OIDC_TOKEN_ENCRYPTION_KEYS and run oidc_rotate_keys. The user stays signed in throughout,
    and afterwards the stored tokens no longer depend on SECRET_KEY."""
    key = Fernet.generate_key().decode()
    row = create_oidc_session(client, user)  # encrypted with the derived key
    access_token = row.access_token
    out = io.StringIO()
    with override_settings(OIDC_TOKEN_ENCRYPTION_KEYS=[key]):
        assert client.get('/api/fresh/', **XHR).status_code == 200  # before the rotation
        call_command('oidc_rotate_keys', stdout=out)
        assert client.get('/api/fresh/', **XHR).status_code == 200  # after it
    assert 're-encrypted 1 sign-in(s); 0 could not be decrypted' in out.getvalue()
    stored = OIDCSession.objects.get(pk=row.pk)
    assert stored.ended_at is None
    with override_settings(OIDC_TOKEN_ENCRYPTION_KEYS=[key], SECRET_KEY='a-later-secret-key-0123456789abcdef',
                           SECRET_KEY_FALLBACKS=[]):
        assert OIDCSession.objects.get(pk=row.pk).access_token == access_token


@pytest.mark.django_db
def test_login_required_decorator_sends_to_the_oidc_login(client):
    response = client.get('/public/decorated/')
    assert response.status_code == 302
    location = urlparse(response['Location'])
    assert location.path == '/oidc/authenticate'
    assert parse_qs(location.query) == {'next': ['/public/decorated/'], 'fail': ['/']}


@pytest.mark.django_db
def test_get_id_claims(user, oidc_op):
    request = RequestFactory().get('/')
    assert get_id_claims(request) == {}
    create_oidc_session(request, user, claims={'email': 'alice@example.test'})
    assert get_id_claims(request)['email'] == 'alice@example.test'


def test_deprecated_middleware_mixin_answers_with_the_contract():
    from oauth2_authcodeflow.middleware import Oauth2MiddlewareMixin

    def check(request):
        raise MiddlewareException('custom 2.x message')
    with pytest.warns(DeprecationWarning):
        middleware = Oauth2MiddlewareMixin(lambda request: HttpResponse('ok'), check_function=check)
    response = middleware(RequestFactory().get('/x', **XHR))
    assert response.status_code == 401
    assert json.loads(response.content)['error'] == 'not_authenticated'  # only contract codes reach clients


@pytest.mark.django_db
def test_attached_access_token_sign_in_ends_with_its_token(user, oidc_op):
    request = RequestFactory().get('/')
    SessionMiddleware(lambda r: HttpResponse()).process_request(request)
    request.user = user
    token = oidc_op.issue_tokens(f'sub-{user.pk}', email=user.email)['access_token']
    row = sessions.attach_access_token(request, user, token, validate_access_token(token))
    assert request.session['_oidc_session_id'] == str(row.pk) and not row.refresh_token_enc
    assert get_access_token(request) == token
    OIDCSession.objects.filter(pk=row.pk).update(access_expires_at=timezone.now() - timedelta(seconds=1))
    with pytest.raises(OIDCSessionEnded, match='no_refresh_token'):
        sessions.ensure_fresh(OIDCSession.objects.get(pk=row.pk))


# --- HTTP client towards the OP -------------------------------------------------------
class _Recorder(http.server.BaseHTTPRequestHandler):
    seen: list = []

    def do_GET(self):
        type(self).seen.append({k.lower(): v for k, v in self.headers.items()})
        if self.path == '/redirect':
            self.send_response(302)
            self.send_header('Location', '/elsewhere')
            self.send_header('Set-Cookie', 'tracking=1; Path=/')
            self.end_headers()
            return
        body = b'{}'
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_op_client_never_follows_redirects_stores_cookies_or_leaks_headers(monkeypatch):
    monkeypatch.setenv('NO_PROXY', '127.0.0.1,localhost')
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        client = OPClient()
        first = client.get(f'{base}/redirect', headers={'Authorization': 'Bearer user-a'})
        assert first.status_code == 302 and len(_Recorder.seen) == 1  # not followed
        assert not list(client.session.cookies)
        client.get(f'{base}/plain')
        assert 'authorization' not in _Recorder.seen[-1] and 'cookie' not in _Recorder.seen[-1]
    finally:
        server.shutdown()
        server.server_close()


def test_http_timeout_forms():
    with override_settings(OIDC_HTTP_TIMEOUT=5):
        assert http_timeout() == 5.0
    with override_settings(OIDC_HTTP_TIMEOUT=(1, 2)):
        assert http_timeout() == (1.0, 2.0)
    with override_settings(OIDC_HTTP_TIMEOUT=(1, 2, 3)):
        with pytest.raises(ValueError):
            http_timeout()


# --- synchronous consumers ------------------------------------------------------------
def _sync_app():
    from channels.auth import AuthMiddlewareStack
    from channels.generic.websocket import WebsocketConsumer
    from channels.routing import URLRouter

    from oauth2_authcodeflow.contrib.channels import (
        OIDCSessionASGIMiddleware,
        SyncOIDCSessionConsumerMixin,
    )

    class Consumer(SyncOIDCSessionConsumerMixin, WebsocketConsumer):
        def connect(self):
            if self.oidc_connect():
                self.accept()

        def disconnect(self, code):
            self.oidc_disconnect()

    return AuthMiddlewareStack(OIDCSessionASGIMiddleware(URLRouter([path('ws/', Consumer.as_asgi())])))


@pytest.mark.django_db(transaction=True)
def test_sync_consumer_contract(user, oidc_op):
    from channels.db import database_sync_to_async

    from test.test_integrations import _Socket
    client = Client()
    row = create_oidc_session(client, user)

    async def scenario():
        anonymous = _Socket(_sync_app())
        assert await anonymous.connect()
        assert await anonymous.close_code() == 4401
        signed_in = _Socket(_sync_app(), client.session.session_key)
        assert await signed_in.connect()
        await database_sync_to_async(sessions.end_session)(row, 'admin')
        assert await signed_in.close_code() == 4401
    async_to_sync(scenario)()


def test_the_pytest_plugin_loads_before_django_is_configured():
    """pytest loads ``pytest11`` plugins before pytest-django configures Django: importing the
    plugin (and the testing package) must not touch settings or models."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    env = {key: value for key, value in os.environ.items() if key != 'DJANGO_SETTINGS_MODULE'}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    code = 'import oauth2_authcodeflow.testing.pytest_plugin, oauth2_authcodeflow.testing as t; assert "FakeOP" in t.__all__'
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-2000:]
