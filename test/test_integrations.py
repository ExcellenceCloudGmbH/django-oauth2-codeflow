"""Bearer tokens, legacy sessions, system checks, DRF, Channels, commands, redaction and test helpers."""
import io
import re
import time
from base64 import urlsafe_b64encode
from datetime import timedelta
from unittest import mock

import jwt
import pytest
from asgiref.sync import async_to_sync
from cryptography.fernet import Fernet
from django.contrib.auth import get_user_model
from django.contrib.sessions.models import Session
from django.core.management import (
    CommandError,
    call_command,
)
from django.db.models.signals import post_save
from django.test import (
    Client,
    RequestFactory,
    override_settings,
)
from django.urls import path
from django.utils import timezone

from oauth2_authcodeflow import (
    checks,
    events,
    sessions,
)
from oauth2_authcodeflow.models import OIDCSession
from oauth2_authcodeflow.testing import (
    create_oidc_session,
    make_signed_token,
)
from oauth2_authcodeflow.tokens import get_access_token

XHR = {'HTTP_SEC_FETCH_MODE': 'cors'}
JWT_RE = re.compile(r'eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.')


# --- bearer ---------------------------------------------------------------------------
@pytest.mark.django_db
def test_bearer_access_token_authenticates_without_a_session(client, oidc_op):
    token = oidc_op.issue_tokens('sub-b', email='bob@example.test', client_roles=['viewer'])['access_token']
    body = client.get('/api/data/', HTTP_AUTHORIZATION=f'Bearer {token}').json()
    assert body['bearer'] is True and body['roles'] == ['viewer']
    assert Session.objects.count() == 0


@pytest.mark.django_db
def test_bearer_rejects_id_tokens_and_symmetric_tokens(client, oidc_op):
    tokens = oidc_op.issue_tokens('sub-b')
    claims = jwt.decode(tokens['access_token'], options={'verify_signature': False})
    forged = jwt.encode(claims, 'test-secret', algorithm='HS256', headers={'kid': oidc_op.kid})
    with override_settings(OIDC_RP_SIGN_ALGOS_ALLOWED=['HS256', 'RS256']):
        for token in (tokens['id_token'], forged):
            response = client.get('/api/data/', HTTP_AUTHORIZATION=f'Bearer {token}')
            assert response.status_code == 401
            assert response['WWW-Authenticate'] == 'Bearer error="invalid_token"'


@pytest.mark.django_db
def test_other_authorization_schemes_are_left_alone(client, oidc_op, caplog):
    with caplog.at_level('WARNING'):
        response = client.get('/api/data/', HTTP_AUTHORIZATION='Api-Key abc.def')
    assert response.status_code == 401 and response['WWW-Authenticate'] == 'Session realm="test-client"'
    assert not [r for r in caplog.records if r.name.startswith('oauth2_authcodeflow')]


@pytest.mark.django_db
def test_bearer_user_is_saved_only_when_claims_change(client, oidc_op):
    saves = []
    post_save.connect(lambda **kw: saves.append(1), sender=get_user_model(), weak=False, dispatch_uid='t-save')
    try:
        token = oidc_op.issue_tokens('sub-c', email='c@example.test')['access_token']
        client.get('/api/data/', HTTP_AUTHORIZATION=f'Bearer {token}')
        client.get('/api/data/', HTTP_AUTHORIZATION=f'Bearer {token}')
    finally:
        post_save.disconnect(dispatch_uid='t-save')
    assert len(saves) == 1


# --- legacy (2.x) sessions ------------------------------------------------------------
def _legacy(client, user, op, **override):
    client.force_login(user, backend='oauth2_authcodeflow.auth.AuthenticationBackend')
    tokens = op.issue_tokens('sub-legacy', email=user.email)
    session = client.session
    values = {
        'oidc_access_token': tokens['access_token'], 'oidc_refresh_token': tokens['refresh_token'],
        'oidc_id_token': tokens['id_token'], 'oidc_access_expires_at': int(time.time()) + 300,
        'oidc_expires_at': int(time.time()) + 86400, 'oidc_op_token_url': op.token_endpoint, 'oidc_op_jwks': {},
    }
    values.update(override)
    for key, value in values.items():
        session[key] = value
    session.save()


@pytest.mark.django_db
def test_legacy_session_is_migrated_once(client, user, oidc_op):
    _legacy(client, user, oidc_op)
    assert client.get('/api/data/', **XHR).status_code == 200
    assert client.get('/api/data/', **XHR).status_code == 200
    row = OIDCSession.objects.get()
    assert row.migrated_from_legacy and row.sub == 'sub-legacy'
    assert 'oidc_access_token' not in client.session and client.session['_oidc_session_id'] == str(row.pk)


@pytest.mark.django_db
def test_expired_legacy_session_is_refreshed(client, user, oidc_op):
    _legacy(client, user, oidc_op, oidc_access_expires_at=int(time.time()) - 10)
    assert client.get('/api/data/', **XHR).status_code == 200
    assert oidc_op.refresh_calls() == 1


@pytest.mark.django_db
def test_garbage_legacy_session_signs_out_cleanly(client, user, oidc_op):
    _legacy(client, user, oidc_op, oidc_access_token=12345)
    response = client.get('/api/data/', **XHR)
    assert response.status_code == 401 and response.json()['error'] == 'session_expired'
    assert not OIDCSession.objects.exists()


# --- system checks --------------------------------------------------------------------
def _ids(messages):
    return [m.id for m in messages]


def test_check_middleware_order():
    wrong = [
        'django.contrib.sessions.middleware.SessionMiddleware',
        'oauth2_authcodeflow.middleware.LoginRequiredMiddleware',
        'oauth2_authcodeflow.middleware.OIDCSessionMiddleware',
        'django.contrib.auth.middleware.AuthenticationMiddleware',
    ]
    with override_settings(MIDDLEWARE=wrong):
        assert set(_ids(checks.check_middleware_order())) == {'oauth2_authcodeflow.E001'}
    assert checks.check_middleware_order() == []


def test_check_algorithms_issuer_distribution():
    with override_settings(OIDC_RP_SIGN_ALGOS_ALLOWED=['HS256', 'RS256']):
        assert _ids(checks.check_algorithms()) == ['oauth2_authcodeflow.E002']
    with override_settings(OIDC_OP_DISCOVERY_DOCUMENT_URL='http://keycloak.default.svc.cluster.local/realms/x/.well-known/openid-configuration',
                           OIDC_OP_ISSUER=''):
        assert _ids(checks.check_internal_issuer()) == ['oauth2_authcodeflow.E003']
    assert checks.check_internal_issuer() == []
    with mock.patch.object(checks.metadata, 'distribution', return_value=object()):
        assert _ids(checks.check_conflicting_distribution()) == ['oauth2_authcodeflow.E004']


def test_check_warnings():
    with override_settings(DEBUG=False, OIDC_TOKEN_ENCRYPTION_KEYS=None):
        assert _ids(checks.check_encryption_keys()) == ['oauth2_authcodeflow.W001']
    with override_settings(OIDC_LOGIN_REQUIRED_EXEMPT_URL_PATTERNS=['/health']):
        assert _ids(checks.check_exempt_patterns()) == ['oauth2_authcodeflow.W002']
    with override_settings(SESSION_COOKIE_AGE=1209600):
        assert _ids(checks.check_session_cookie_age()) == ['oauth2_authcodeflow.W003']


# --- Django REST framework ------------------------------------------------------------
@pytest.mark.django_db
def test_drf_unauthenticated_is_401_with_code(client):
    with override_settings(OIDC_LOGIN_REQUIRED_EXEMPT_URL_PATTERNS=[r'^/api/drf']):
        response = client.get('/api/drf/')
    assert response.status_code == 401 and response.json()['code'] == 'not_authenticated'
    assert response['WWW-Authenticate']


@pytest.mark.django_db
def test_drf_permission_and_csrf_codes(user, oidc_op):
    client = Client(enforce_csrf_checks=True)
    create_oidc_session(client, user)
    assert client.get('/api/drf-denied/').json()['code'] == 'permission_denied'
    response = client.post('/api/drf/')
    assert response.status_code == 403 and response.json()['code'] == 'csrf_failed'


@pytest.mark.django_db
def test_drf_maps_library_exceptions(client, user, oidc_op):
    create_oidc_session(client, user)
    response = client.get('/api/drf-raise/', {'kind': 'ended'})
    assert response.status_code == 401 and response.json()['error'] == 'session_expired'
    response = client.get('/api/drf-raise/', {'kind': 'unavailable'})
    assert response.status_code == 503 and response['Retry-After'] == '3'


# --- Channels -------------------------------------------------------------------------
def _ws_app(revalidate=0):
    from channels.auth import AuthMiddlewareStack
    from channels.generic.websocket import AsyncWebsocketConsumer
    from channels.routing import URLRouter

    from oauth2_authcodeflow.contrib.channels import (
        OIDCSessionASGIMiddleware,
        OIDCSessionConsumerMixin,
    )

    class Consumer(OIDCSessionConsumerMixin, AsyncWebsocketConsumer):
        oidc_revalidate_seconds = revalidate

        async def connect(self):
            if await self.oidc_connect():
                await self.accept()

        async def disconnect(self, code):
            await self.oidc_disconnect()

    return AuthMiddlewareStack(OIDCSessionASGIMiddleware(URLRouter([path('ws/', Consumer.as_asgi())])))


class _Socket:
    """Minimal websocket client on asgiref's communicator (channels.testing needs daphne)."""

    def __init__(self, app, session_key=None):
        from asgiref.testing import ApplicationCommunicator
        headers = [(b'cookie', f'sessionid={session_key}'.encode())] if session_key else []
        scope = {'type': 'websocket', 'path': '/ws/', 'raw_path': b'/ws/', 'query_string': b'', 'headers': headers,
                 'subprotocols': [], 'client': ['127.0.0.1', 50000], 'server': ['testserver', 80]}
        self.communicator = ApplicationCommunicator(app, scope)

    async def connect(self):
        await self.communicator.send_input({'type': 'websocket.connect'})
        return (await self.communicator.receive_output(2))['type'] == 'websocket.accept'

    async def close_code(self, timeout=2):
        message = await self.communicator.receive_output(timeout)
        assert message['type'] == 'websocket.close', message
        return message.get('code')


def _communicator(app, session_key=None):
    return _Socket(app, session_key)


@pytest.mark.django_db(transaction=True)
def test_anonymous_socket_is_accepted_then_closed_4401():
    async def scenario():
        communicator = _communicator(_ws_app())
        assert await communicator.connect()
        assert await communicator.close_code() == 4401
    async_to_sync(scenario)()


@pytest.mark.django_db(transaction=True)
def test_socket_closes_when_the_sign_in_ends(user, oidc_op):
    from channels.db import database_sync_to_async
    client = Client()
    row = create_oidc_session(client, user)

    async def scenario():
        communicator = _communicator(_ws_app(), client.session.session_key)
        assert await communicator.connect()
        await database_sync_to_async(sessions.end_session)(row, 'backchannel_logout')
        assert await communicator.close_code() == 4401
    async_to_sync(scenario)()


@pytest.mark.django_db(transaction=True)
def test_socket_revalidation_closes_ended_sign_ins(user, oidc_op):
    client = Client()
    row = create_oidc_session(client, user)

    async def scenario():
        from channels.db import database_sync_to_async
        communicator = _communicator(_ws_app(revalidate=0.2), client.session.session_key)
        assert await communicator.connect()
        await database_sync_to_async(OIDCSession.objects.filter(pk=row.pk).update)(ended_at=timezone.now(), end_reason='admin')
        assert await communicator.close_code(timeout=3) == 4401
    async_to_sync(scenario)()


# --- management commands --------------------------------------------------------------
def test_oidc_urls():
    out = io.StringIO()
    err = io.StringIO()
    call_command('oidc_urls', 'app.example.com', '--secure', stdout=out, stderr=err)
    assert out.getvalue().splitlines() == [
        'redirect_url: https://app.example.com/oidc/callback',
        'post_logout_redirect_url: https://app.example.com/oidc/callback',
        'backchannel_logout_url: https://app.example.com/oidc/backchannel-logout',
    ]
    assert 'ALLOWED_HOSTS' in err.getvalue()  # the operator is told Django would refuse this host
    with override_settings(ALLOWED_HOSTS=['.example.com']):
        err = io.StringIO()
        call_command('oidc_urls', 'app.example.com', stdout=io.StringIO(), stderr=err)
        assert err.getvalue() == ''
    with override_settings(ROOT_URLCONF='test.no_oidc_urls'):
        with pytest.raises(CommandError, match='oidc_callback'):
            call_command('oidc_urls', 'app.example.com', stdout=io.StringIO(), stderr=io.StringIO())


@pytest.mark.django_db
def test_oidc_check(oidc_op):
    out = io.StringIO()
    call_command('oidc_check', stdout=out, stderr=io.StringIO())
    assert 'OIDC configuration OK' in out.getvalue()
    with override_settings(OIDC_OP_ISSUER='https://other.example.test/realms/test'):
        with pytest.raises(CommandError):
            call_command('oidc_check', stdout=io.StringIO(), stderr=io.StringIO())


@pytest.mark.django_db
def test_oidc_cleanup(user, oidc_op, django_capture_on_commit_callbacks):
    expired = create_oidc_session(Client(), user)
    OIDCSession.objects.filter(pk=expired.pk).update(refresh_expires_at=timezone.now() - timedelta(minutes=1))
    old = create_oidc_session(Client(), user)
    OIDCSession.objects.filter(pk=old.pk).update(ended_at=timezone.now() - timedelta(days=30), end_reason='logout')
    keep = create_oidc_session(Client(), user)
    with django_capture_on_commit_callbacks(execute=True):
        call_command('oidc_cleanup', stdout=io.StringIO())
    assert OIDCSession.objects.get(pk=expired.pk).end_reason == 'refresh_expired'
    assert not OIDCSession.objects.filter(pk=old.pk).exists()
    assert OIDCSession.objects.get(pk=keep.pk).ended_at is None


@pytest.mark.django_db
def test_oidc_rotate_keys(user, oidc_op):
    old_key, new_key = Fernet.generate_key().decode(), Fernet.generate_key().decode()
    with override_settings(OIDC_TOKEN_ENCRYPTION_KEYS=[old_key]):
        row = create_oidc_session(Client(), user)
        token = OIDCSession.objects.get(pk=row.pk).access_token
    with override_settings(OIDC_TOKEN_ENCRYPTION_KEYS=[new_key, old_key]):
        call_command('oidc_rotate_keys', stdout=io.StringIO())
    with override_settings(OIDC_TOKEN_ENCRYPTION_KEYS=[new_key]):
        assert OIDCSession.objects.get(pk=row.pk).access_token == token


@pytest.mark.django_db
def test_purge_blacklisted_tokens_is_deprecated():
    with pytest.warns(DeprecationWarning):
        call_command('purge_blacklisted_tokens', stdout=io.StringIO())


# --- redaction ------------------------------------------------------------------------
def test_redact():
    token = make_signed_token({'sub': 'x'})
    assert 'eyJ' not in events.redact(f'bad token {token} here')


@pytest.mark.django_db
def test_no_token_ever_reaches_logs_or_responses(client, user, oidc_op, caplog):
    bodies = []
    with caplog.at_level('DEBUG'):
        create_oidc_session(client, user, access_expires_in=1)
        oidc_op.script_token((400, {'error': 'invalid_grant', 'error_description': f'bad {oidc_op.issue_tokens()["refresh_token"]}'}))
        bodies.append(client.get('/api/data/', **XHR).content)
        bodies.append(client.get('/api/data/', HTTP_AUTHORIZATION=f'Bearer {oidc_op.issue_tokens()["id_token"]}').content)
        bodies.append(client.post('/oidc/backchannel-logout', {'logout_token': oidc_op.logout_token(sid='s', events=False)}).content)
    assert not JWT_RE.search(caplog.text)
    assert not any(JWT_RE.search(body.decode()) for body in bodies)


# --- test helpers ---------------------------------------------------------------------
@pytest.mark.django_db
def test_create_oidc_session_with_a_request(user, oidc_op):
    request = RequestFactory().get('/')
    row = create_oidc_session(request, user, client_roles=['admin'])
    assert get_access_token(request) == row.access_token
    assert request.oidc.client_roles() == ['admin']


def test_make_signed_token_uses_the_fake_op_key():
    token = make_signed_token({'sub': 'x'})
    assert jwt.get_unverified_header(token)['kid'] == 'fake-op-key'
    assert urlsafe_b64encode(b'x')
