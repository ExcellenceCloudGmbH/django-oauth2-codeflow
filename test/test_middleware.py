"""The HTTP auth contract through the middlewares."""
import pytest
from django.test import override_settings

from oauth2_authcodeflow.models import OIDCSession
from oauth2_authcodeflow.testing import create_oidc_session

pytestmark = pytest.mark.django_db

NAV = {'HTTP_ACCEPT': 'text/html', 'HTTP_SEC_FETCH_MODE': 'navigate', 'HTTP_SEC_FETCH_DEST': 'document'}
FRAME = {'HTTP_ACCEPT': 'text/html', 'HTTP_SEC_FETCH_MODE': 'navigate', 'HTTP_SEC_FETCH_DEST': 'iframe'}
XHR = {'HTTP_ACCEPT': 'application/json', 'HTTP_SEC_FETCH_MODE': 'cors'}


def test_anonymous_navigation_is_redirected_to_login(client):
    response = client.get('/protected/', **NAV)
    assert response.status_code == 302
    assert response['Location'] == '/oidc/authenticate?next=%2Fprotected%2F'
    assert response['Cache-Control'] == 'no-store'


@pytest.mark.parametrize('path,headers', [
    ('/protected/', XHR),                                # fetch
    ('/protected/', {'HTTP_X_REQUESTED_WITH': 'XMLHttpRequest'}),
    ('/api/data/', NAV),                                 # API paths always get JSON
    ('/protected/', {}),                                 # no browser hints
])
def test_anonymous_background_requests_get_json_401(client, path, headers):
    response = client.get(path, **headers)
    assert response.status_code == 401
    assert response.json() == {'error': 'not_authenticated', 'login_url': '/oidc/authenticate'}
    assert response['WWW-Authenticate'] == 'Session realm="test-client"'


def test_anonymous_frame_gets_a_top_level_sign_in_link(client):
    response = client.get('/protected/', **FRAME)
    assert response.status_code == 401 and b'target="_top"' in response.content


def test_exempt_paths_pass(client):
    assert client.get('/public/', **XHR).status_code == 200
    assert client.get('/health').status_code == 200


def test_signed_in_request_has_context_and_roles(client, user, oidc_op):
    create_oidc_session(client, user, client_roles=['admin'])
    body = client.get('/api/data/', **XHR).json()
    assert body == {'user': 'alice', 'has_oidc': True, 'bearer': False, 'roles': ['admin']}


def test_anonymous_request_never_destroys_a_pending_login(client, oidc_op):
    client.get('/oidc/authenticate', {'next': '/protected/'})
    pending = dict(client.session['_oidc_pending'])
    assert client.get('/api/data/', **XHR).status_code == 401
    assert client.get('/protected/', **NAV).status_code == 302
    assert set(client.session['_oidc_pending']) >= set(pending)


def test_session_ended_by_the_op_gives_session_expired_and_signs_out(client, user, oidc_op):
    create_oidc_session(client, user, access_expires_in=1)
    oidc_op.script_token((400, {'error': 'invalid_grant'}))
    response = client.get('/api/data/', **XHR)
    assert response.status_code == 401 and response.json()['error'] == 'session_expired'
    assert client.get('/api/data/', **XHR).json()['error'] == 'not_authenticated'  # signed out


def test_unavailable_op_with_an_expired_token_answers_503(client, user, oidc_op):
    create_oidc_session(client, user, access_expires_in=1)
    oidc_op.script_token((503, 'down'))
    response = client.get('/api/data/', **XHR)
    assert response.status_code == 503
    assert response.json()['error'] == 'auth_unavailable' and int(response['Retry-After']) >= 1
    navigation = client.get('/protected/', **NAV)  # breaker still closed: one more transient failure
    assert navigation.status_code in (503, 200)
    assert OIDCSession.objects.get().ended_at is None


def test_unavailable_op_with_a_valid_token_is_invisible(client, user, oidc_op):
    create_oidc_session(client, user, access_expires_in=30)
    oidc_op.script_token((503, 'down'))
    assert client.get('/api/data/', **XHR).status_code == 200


def test_view_level_exceptions_follow_the_contract(client, user, oidc_op):
    create_oidc_session(client, user)
    response = client.get('/api/raise/', {'kind': 'unavailable'}, **XHR)
    assert response.status_code == 503 and response['Retry-After'] == '7'
    response = client.get('/api/raise/', {'kind': 'ended'}, **XHR)
    assert response.status_code == 401 and response.json()['error'] == 'session_expired'


def test_get_access_token_min_ttl_refreshes(client, user, oidc_op):
    create_oidc_session(client, user, access_expires_in=120)
    body = client.get('/api/fresh/', {'min_ttl': '200'}, **XHR).json()
    assert body['has_token'] and oidc_op.refresh_calls() == 1


def test_logins_of_other_backends_pass_untouched(client, user):
    client.force_login(user, backend='django.contrib.auth.backends.ModelBackend')
    body = client.get('/api/data/', **XHR).json()
    assert body['user'] == 'alice' and body['has_oidc'] is False


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_deprecated_aliases_run_once(client, user, oidc_op):
    middleware = [
        'django.contrib.sessions.middleware.SessionMiddleware',
        'django.contrib.auth.middleware.AuthenticationMiddleware',
        'oauth2_authcodeflow.middleware.RefreshSessionMiddleware',
        'oauth2_authcodeflow.middleware.RefreshAccessTokenMiddleware',
        'oauth2_authcodeflow.middleware.OIDCSessionMiddleware',
        'oauth2_authcodeflow.middleware.LoginRequiredMiddleware',
    ]
    with override_settings(MIDDLEWARE=middleware):
        create_oidc_session(client, user, access_expires_in=1)
        assert client.get('/api/data/', **XHR).status_code == 200
    assert oidc_op.refresh_calls() == 1


def test_lazy_mode_refreshes_only_on_demand(client, user, oidc_op):
    create_oidc_session(client, user, access_expires_in=30)
    with override_settings(OIDC_REFRESH_MODE='lazy'):
        assert client.get('/api/data/', **XHR).status_code == 200
        assert oidc_op.refresh_calls() == 0
        assert client.get('/api/fresh/', {'min_ttl': '60'}, **XHR).json()['has_token']
    assert oidc_op.refresh_calls() == 1
