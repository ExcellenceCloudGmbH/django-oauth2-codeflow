"""Sign-in, callback, logout and back-channel logout views."""
import hashlib
from base64 import urlsafe_b64encode
from urllib.parse import (
    parse_qs,
    urlparse,
)

import pytest
from django.contrib.auth import (
    get_user_model,
    user_logged_in,
)
from django.test import (
    Client,
    override_settings,
)

from oauth2_authcodeflow import signals
from oauth2_authcodeflow.conf import get_default_django_username
from oauth2_authcodeflow.models import OIDCSession
from oauth2_authcodeflow.testing import create_oidc_session
from oauth2_authcodeflow.tokens import get_access_token

pytestmark = pytest.mark.django_db

XHR = {'HTTP_SEC_FETCH_MODE': 'cors'}


def _start(client, **params):
    response = client.get('/oidc/authenticate', {'next': '/protected/', **params})
    assert response.status_code == 302, response.content
    return {k: v[0] for k, v in parse_qs(urlparse(response['Location']).query).items()}


def _finish(client, op, auth, sub='sub-new', email='new@example.test', **extra):
    code = op.authorization_code(sub, nonce=auth['nonce'], email=email, **extra)
    return client.get('/oidc/callback', {'code': code, 'state': auth['state']})


# --- sign-in --------------------------------------------------------------------------
def test_authorization_request_uses_pkce_state_and_nonce(client, oidc_op):
    auth = _start(client, prompt='none', login_hint='x@example.test')
    assert auth['response_type'] == 'code' and auth['client_id'] == 'test-client'
    assert auth['code_challenge_method'] == 'S256' and auth['state'] and auth['nonce']
    assert auth['redirect_uri'] == 'http://testserver/oidc/callback'
    assert auth['prompt'] == 'none' and auth['login_hint'] == 'x@example.test'
    assert auth['scope'] == 'openid email profile'


def test_unknown_prompt_values_are_ignored(client, oidc_op):
    assert 'prompt' not in _start(client, prompt='evil')


@pytest.mark.parametrize('field', ['next', 'fail'])
def test_open_redirects_are_refused(client, oidc_op, field):
    response = client.get('/oidc/authenticate', {field: 'https://evil.example/'})
    assert response.status_code == 400


def test_full_sign_in(client, oidc_op):
    auth = _start(client)
    response = _finish(client, oidc_op, auth)
    assert response.status_code == 302 and response['Location'] == '/protected/'
    row = OIDCSession.objects.get()
    assert row.sub == 'sub-new' and row.sid == 'sid-sub-new' and row.claims['email'] == 'new@example.test'
    assert client.session['_oidc_session_id'] == str(row.pk) and row.django_session_key == client.session.session_key
    call = oidc_op.token_calls[0]
    expected = urlsafe_b64encode(hashlib.sha256(call['code_verifier'].encode()).digest()).decode().rstrip('=')
    assert expected == auth['code_challenge'] and call['client_secret'] == 'test-secret'
    user = get_user_model().objects.get()
    assert user.email == 'new@example.test' and not user.has_usable_password()
    assert client.get('/protected/').status_code == 200


def test_two_parallel_sign_ins_both_complete(client, oidc_op):
    first, second = _start(client), _start(client)
    assert _finish(client, oidc_op, first).status_code == 302
    assert _finish(client, oidc_op, second).status_code == 302
    rows = OIDCSession.objects.order_by('created_at')
    assert [r.end_reason for r in rows] == ['superseded', '']


def test_pending_logins_are_capped(client, oidc_op):
    states = [_start(client) for _ in range(6)]
    response = _finish(client, oidc_op, states[0])
    assert response.status_code == 400 and b'invalid_state' in response.content
    assert _finish(client, oidc_op, states[-1]).status_code == 302


def test_callback_error_never_signs_anyone_out(client, user, oidc_op):
    row = create_oidc_session(client, user)
    auth = _start(client, prompt='none')
    response = client.get('/oidc/callback', {'error': 'login_required', 'state': auth['state']})
    assert response.status_code == 400 and b'login_required' in response.content
    assert client.get('/api/data/', **XHR).status_code == 200
    assert OIDCSession.objects.get(pk=row.pk).ended_at is None


def test_failure_redirects_to_fail_url(client, oidc_op):
    auth = _start(client, fail='/public/')
    response = client.get('/oidc/callback', {'error': 'access_denied', 'state': auth['state']})
    assert response['Location'] == '/public/?error=access_denied'


def test_nonce_mismatch_is_rejected(client, oidc_op):
    auth = _start(client)
    code = oidc_op.authorization_code('sub-new', nonce='wrong-nonce', email='n@example.test')
    response = client.get('/oidc/callback', {'code': code, 'state': auth['state']})
    assert response.status_code == 400 and b'invalid_id_token' in response.content
    assert not OIDCSession.objects.exists()


def test_unavailable_op_during_code_exchange(client, oidc_op):
    auth = _start(client, fail='/public/')
    oidc_op.script_token((503, 'down'))
    response = client.get('/oidc/callback', {'code': 'c', 'state': auth['state']})
    assert response['Location'] == '/public/?error=auth_unavailable'


def test_existing_user_keeps_password_and_other_sessions(client, oidc_op):
    email = 'carol@example.test'
    user = get_user_model().objects.create_user(get_default_django_username({'email': email}), email, 'pw-carol-123')
    password = user.password
    other_device = Client()
    other_device.force_login(user, backend='django.contrib.auth.backends.ModelBackend')
    assert _finish(client, oidc_op, _start(client), sub='carol', email=email).status_code == 302
    user.refresh_from_db()
    assert user.password == password
    assert other_device.get('/protected/').status_code == 200  # still signed in


def test_inactive_and_unknown_users(client, oidc_op):
    email = 'dave@example.test'
    get_user_model().objects.create_user(get_default_django_username({'email': email}), email, is_active=False)
    response = _finish(client, oidc_op, _start(client), sub='dave', email=email)
    assert response.status_code == 400 and b'user_inactive' in response.content
    with override_settings(OIDC_CREATE_USER=False):
        response = _finish(client, oidc_op, _start(client), sub='erin', email='erin@example.test')
    assert b'user_not_allowed' in response.content


def test_user_logged_in_receivers_can_read_the_access_token(client, oidc_op):
    seen = []

    def receiver(sender, request, user, **kwargs):
        seen.append(get_access_token(request))
    user_logged_in.connect(receiver, dispatch_uid='t-login')
    try:
        _finish(client, oidc_op, _start(client))
    finally:
        user_logged_in.disconnect(dispatch_uid='t-login')
    assert seen and seen[0].startswith('eyJ')


# --- logout ---------------------------------------------------------------------------
def test_local_logout_ends_the_sign_in_without_revoking(client, user, oidc_op, django_capture_on_commit_callbacks):
    row = create_oidc_session(client, user)
    with django_capture_on_commit_callbacks(execute=True):
        response = client.get('/oidc/logout', {'next': '/public/'})
    assert response['Location'] == '/public/'
    assert OIDCSession.objects.get(pk=row.pk).end_reason == 'logout'
    assert oidc_op.revoke_calls == []
    assert client.get('/api/data/', **XHR).status_code == 401


def test_revocation_on_logout_when_enabled(client, user, oidc_op, django_capture_on_commit_callbacks):
    create_oidc_session(client, user)
    with override_settings(OIDC_REVOKE_ON_LOGOUT=True), django_capture_on_commit_callbacks(execute=True):
        client.get('/oidc/logout')
    assert len(oidc_op.revoke_calls) == 1


def test_total_logout_sends_id_token_hint_and_returns(client, user, oidc_op):
    row = create_oidc_session(client, user)
    id_token = row.id_token
    response = client.get('/oidc/total_logout', {'next': '/public/'})
    url = urlparse(response['Location'])
    params = {k: v[0] for k, v in parse_qs(url.query).items()}
    assert f'{url.scheme}://{url.netloc}{url.path}' == oidc_op.end_session_endpoint
    assert params['id_token_hint'] == id_token and params['client_id'] == 'test-client'
    assert params['post_logout_redirect_uri'] == 'http://testserver/oidc/callback'
    assert OIDCSession.objects.get(pk=row.pk).end_reason == 'total_logout'
    back = client.get('/oidc/callback', {'state': params['state']})
    assert back['Location'] == '/public/'


def test_logout_refuses_foreign_redirects(client, user, oidc_op):
    create_oidc_session(client, user)
    assert client.get('/oidc/total_logout', {'next': 'https://evil.example/'}).status_code == 400


# --- back-channel logout --------------------------------------------------------------
def test_backchannel_logout_by_sid(user, oidc_op, django_user_model, django_capture_on_commit_callbacks):
    bob = django_user_model.objects.create_user('bob', 'bob@example.test', 'pw-bob-123')
    a = create_oidc_session(Client(), user, sid='kc-1')
    b = create_oidc_session(Client(), bob, sid='kc-1')
    c = create_oidc_session(Client(), bob, sid='kc-2')
    received = []
    signals.oidc_backchannel_logout_received.connect(lambda **kw: received.append(kw), weak=False, dispatch_uid='t-bcl')
    try:
        with django_capture_on_commit_callbacks(execute=True):
            response = Client(enforce_csrf_checks=True).post('/oidc/backchannel-logout', {'logout_token': oidc_op.logout_token(sid='kc-1')})
    finally:
        signals.oidc_backchannel_logout_received.disconnect(dispatch_uid='t-bcl')
    assert response.status_code == 200 and response['Cache-Control'] == 'no-store'
    assert [OIDCSession.objects.get(pk=r.pk).end_reason for r in (a, b, c)] == ['backchannel_logout', 'backchannel_logout', '']
    assert received[0]['matched'] == 2 and received[0]['sid'] == 'kc-1'


def test_backchannel_logout_by_sub_unknown_and_replay(client, user, oidc_op):
    row = create_oidc_session(client, user)
    received = []
    signals.oidc_backchannel_logout_received.connect(lambda **kw: received.append(kw['matched']), weak=False, dispatch_uid='t-bcl2')
    try:
        unknown = oidc_op.logout_token(sid='nobody')
        assert client.post('/oidc/backchannel-logout', {'logout_token': unknown}).status_code == 200
        token = oidc_op.logout_token(sub=row.sub)
        assert client.post('/oidc/backchannel-logout', {'logout_token': token}).status_code == 200
        assert client.post('/oidc/backchannel-logout', {'logout_token': token}).status_code == 200  # replay
    finally:
        signals.oidc_backchannel_logout_received.disconnect(dispatch_uid='t-bcl2')
    assert received == [0, 1]
    assert OIDCSession.objects.get(pk=row.pk).end_reason == 'backchannel_logout'


@pytest.mark.parametrize('kwargs', [{'events': False, 'sid': 's'}, {'sid': 's', 'extra': {'nonce': 'n'}}, {'sid': 's', 'extra': {'aud': 'x'}}])
def test_backchannel_logout_rejects_invalid_tokens(client, oidc_op, kwargs):
    response = client.post('/oidc/backchannel-logout', {'logout_token': oidc_op.logout_token(**kwargs)})
    assert response.status_code == 400 and response.json() == {'error': 'invalid_request'}


def test_backchannel_logout_method_and_switch(client, oidc_op):
    assert client.get('/oidc/backchannel-logout').status_code == 405
    assert client.post('/oidc/backchannel-logout').status_code == 400
    with override_settings(OIDC_BACKCHANNEL_LOGOUT_ENABLED=False):
        assert client.post('/oidc/backchannel-logout', {'logout_token': oidc_op.logout_token(sid='s')}).status_code == 404
