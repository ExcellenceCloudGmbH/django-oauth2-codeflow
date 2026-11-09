"""The library against a real Keycloak 26 (nightly CI job; locally with Docker).

Run: ``pytest test/keycloak -m keycloak --ds=test.keycloak.settings_kc``

Intent (plan section 6.1): sign-in and refresh through the internal URL keep the public
issuer; an SSO session ended in Keycloak ends the sign-in at the next refresh
(``invalid_grant``); Keycloak's back-channel logout reaches the app and ends the sign-in;
an issuer mismatch at the token endpoint is a configuration error (503), never a logout;
RP-initiated logout returns to the app; Keycloak access tokens work as bearer tokens.
"""
import html
import re
import time
from datetime import timedelta
from urllib.parse import urlparse

import jwt
import pytest
import requests
from django.conf import settings
from django.utils import timezone

from oauth2_authcodeflow.breaker import reset_breakers
from oauth2_authcodeflow.models import OIDCSession
from oauth2_authcodeflow.provider import get_provider

pytestmark = [pytest.mark.keycloak, pytest.mark.django_db(transaction=True), pytest.mark.timeout(300)]

XHR = {'Sec-Fetch-Mode': 'cors', 'Accept': 'application/json'}


def _app(live_server):
    return f'http://localhost:{urlparse(live_server.url).port}'


class _Browser(requests.Session):
    """Treats http://localhost as a secure context like browsers do: Keycloak sets Secure
    cookies there, which requests would otherwise never send back over http."""

    def send(self, request, **kwargs):
        response = super().send(request, **kwargs)
        for cookie in self.cookies:
            cookie.secure = False
        return response


def _browser():
    browser = _Browser()
    browser.trust_env = False
    return browser


def _sign_in(app, username='alice', password='alice-password'):
    """The authorization code flow as a browser does it: app, Keycloak login form, callback."""
    browser = _browser()
    start = browser.get(f'{app}/oidc/authenticate', params={'next': '/protected/'}, allow_redirects=False, timeout=15)
    assert start.status_code == 302, start.text[:300]
    assert start.headers['Location'].startswith(settings.OIDC_OP_ISSUER)  # the browser goes to the public URL
    page = browser.get(start.headers['Location'], timeout=15)
    form = re.search(r'<form[^>]*\bid="kc-form-login"[^>]*>', page.text)
    assert page.status_code == 200 and form, page.text[:500]
    action = html.unescape(re.search(r'\baction="([^"]+)"', form.group(0)).group(1))
    answer = browser.post(action, data={'username': username, 'password': password, 'credentialId': ''},
                          allow_redirects=False, timeout=15)
    assert answer.status_code == 302, answer.text[:500]
    callback = browser.get(answer.headers['Location'], allow_redirects=False, timeout=30)
    assert callback.status_code == 302 and callback.headers['Location'] == '/protected/', (callback.status_code, callback.text[:500])
    return browser


def _expire(row):
    OIDCSession.objects.filter(pk=row.pk).update(access_expires_at=timezone.now() - timedelta(seconds=10))


def _wait_for(predicate, timeout):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, 'condition not reached'
        time.sleep(0.2)


def test_k1_sign_in_and_refresh_through_the_internal_url_keep_the_public_issuer(live_server, kc):
    app = _app(live_server)
    browser = _sign_in(app)
    row = OIDCSession.objects.get()
    assert row.issuer == settings.OIDC_OP_ISSUER and row.sid
    metadata = get_provider().metadata()
    assert metadata.issuer == settings.OIDC_OP_ISSUER
    assert metadata.token_endpoint.startswith(settings.KC_INTERNAL_URL)  # back-channel calls stay internal
    assert browser.get(f'{app}/api/data/', headers=XHR, timeout=15).json()['has_oidc'] is True
    _expire(row)
    response = browser.get(f'{app}/api/data/', headers=XHR, timeout=30)
    assert response.status_code == 200, response.text
    fresh = OIDCSession.objects.get(pk=row.pk)
    assert fresh.version == row.version + 1 and fresh.ended_at is None
    assert jwt.decode(fresh.access_token, options={'verify_signature': False})['iss'] == settings.OIDC_OP_ISSUER


def test_k2_an_sso_session_ended_in_keycloak_ends_the_sign_in_at_the_next_refresh(live_server, kc):
    app = _app(live_server)
    browser = _sign_in(app)
    row = OIDCSession.objects.get()
    kc.logout_user('alice')  # no back-channel URL configured: only the refresh can notice
    _expire(row)
    response = browser.get(f'{app}/api/data/', headers=XHR, timeout=30)
    assert response.status_code == 401 and response.json()['error'] == 'session_expired', response.text
    assert OIDCSession.objects.get(pk=row.pk).end_reason == 'refresh_invalid_grant'


def test_k3_keycloak_backchannel_logout_reaches_the_app(live_server, kc):
    app = _app(live_server)
    port = urlparse(live_server.url).port
    kc.update_client(**{'backchannel.logout.url': f'http://host.docker.internal:{port}/oidc/backchannel-logout'})
    browser = _sign_in(app)
    row = OIDCSession.objects.get()
    kc.logout_user('alice')
    _wait_for(lambda: OIDCSession.objects.get(pk=row.pk).ended_at is not None, timeout=30)
    assert OIDCSession.objects.get(pk=row.pk).end_reason == 'backchannel_logout'
    assert browser.get(f'{app}/api/data/', headers=XHR, timeout=15).status_code == 401


def test_k4_an_issuer_mismatch_at_the_token_endpoint_is_configuration_not_logout(live_server, kc):
    app = _app(live_server)
    browser = _sign_in(app)
    row = OIDCSession.objects.get()
    kc.set_frontend_url('http://kc-elsewhere.example.test:18080')  # Keycloak now expects another issuer
    try:
        _expire(row)
        response = browser.get(f'{app}/api/data/', headers=XHR, timeout=30)
    finally:
        kc.set_frontend_url(None)
    assert response.status_code == 503 and response.json()['error'] == 'auth_unavailable', response.text
    assert OIDCSession.objects.get(pk=row.pk).ended_at is None
    reset_breakers()  # the configuration was fixed: the next refresh works and nobody signed in again
    assert browser.get(f'{app}/api/data/', headers=XHR, timeout=30).status_code == 200
    assert OIDCSession.objects.get(pk=row.pk).version == row.version + 1


def test_k5_rp_initiated_logout_ends_the_sso_session_and_returns_to_the_app(live_server, kc):
    app = _app(live_server)
    browser = _sign_in(app)
    assert len(kc.user_sessions('alice')) == 1
    start = browser.get(f'{app}/oidc/total_logout', params={'next': '/public/'}, allow_redirects=False, timeout=15)
    assert start.status_code == 302 and 'id_token_hint=' in start.headers['Location']
    back = browser.get(start.headers['Location'], allow_redirects=False, timeout=15)
    assert back.status_code == 302 and back.headers['Location'].startswith(f'{app}/oidc/callback'), back.text[:500]
    done = browser.get(back.headers['Location'], allow_redirects=False, timeout=15)
    assert done.status_code == 302 and done.headers['Location'] == '/public/'
    assert kc.user_sessions('alice') == []
    assert OIDCSession.objects.get().end_reason == 'total_logout'


def test_k6_keycloak_access_tokens_are_bearer_tokens_and_id_tokens_are_not(live_server, kc):
    app = _app(live_server)
    tokens = kc.password_grant('alice', 'alice-password')
    browser = _browser()
    response = browser.get(f'{app}/api/data/', headers={'Authorization': f"Bearer {tokens['access_token']}"}, timeout=15)
    assert response.status_code == 200 and response.json()['bearer'] is True, response.text
    response = browser.get(f'{app}/api/data/', headers={'Authorization': f"Bearer {tokens['id_token']}"}, timeout=15)
    assert response.status_code == 401
