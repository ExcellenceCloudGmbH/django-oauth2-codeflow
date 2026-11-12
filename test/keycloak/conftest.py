"""Keycloak 26 for the integration suite.

The suite starts ``quay.io/keycloak/keycloak:26.7.5`` in Docker as ``kc-codeflow-3`` on
127.0.0.1:18080 with realm.json imported, and removes it afterwards (OIDC_KC_KEEP=1 keeps
it). OIDC_KC_EXTERNAL=1 uses an already running Keycloak at KC_PUBLIC_URL instead. Without
Docker the suite is skipped; OIDC_KC_REQUIRED=1 (CI) makes that an error.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests

IMAGE = os.environ.get('OIDC_KC_IMAGE', 'quay.io/keycloak/keycloak:26.7.5')
CONTAINER = os.environ.get('OIDC_KC_CONTAINER', 'kc-codeflow-3')
REALM_FILE = Path(__file__).resolve().with_name('realm.json')
CLIENT_ID = 'codeflow-test'

# Keycloak (in a container) reaches the live server through host.docker.internal
os.environ.setdefault('DJANGO_LIVE_TEST_SERVER_ADDRESS', '0.0.0.0')
os.environ['NO_PROXY'] = ','.join(dict.fromkeys(
    [h for h in os.environ.get('NO_PROXY', '').split(',') if h] + ['127.0.0.1', 'localhost', 'host.docker.internal']))


def _docker(*args, check=True):
    return subprocess.run(['docker', *args], capture_output=True, text=True, check=check, timeout=300)


def _unavailable_reason():
    from django.conf import settings
    if not getattr(settings, 'KEYCLOAK_SUITE', False):
        return 'the Keycloak suite runs with --ds=test.keycloak.settings_kc'
    if os.environ.get('OIDC_KC_EXTERNAL'):
        return None
    if shutil.which('docker') is None:
        return 'docker is not installed'
    try:
        info = _docker('info', '--format', '{{.ServerVersion}}', check=False)
    except subprocess.TimeoutExpired:
        return 'the Docker daemon does not answer'
    if info.returncode != 0:
        return f'the Docker daemon is not running: {info.stderr.strip()[:200]}'
    return None


def pytest_collection_modifyitems(config, items):
    kc_items = [item for item in items if item.get_closest_marker('keycloak')]
    if not kc_items:
        return
    reason = _unavailable_reason()
    if reason is None:
        return
    if os.environ.get('OIDC_KC_REQUIRED'):
        raise pytest.UsageError(reason)
    for item in kc_items:
        item.add_marker(pytest.mark.skip(reason=reason))


class Keycloak:
    """Admin calls against the local test container only."""

    def __init__(self, public_url, internal_url, realm):
        self.public_url, self.internal_url, self.realm = public_url, internal_url, realm
        self.http = requests.Session()
        self.http.trust_env = False

    def wait_ready(self, timeout):
        deadline = time.monotonic() + timeout
        url = f'{self.internal_url}/realms/{self.realm}/.well-known/openid-configuration'
        while True:
            try:
                if self.http.get(url, timeout=3).status_code == 200:
                    return
            except requests.RequestException:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f'Keycloak not ready after {timeout}s: {url}')
            time.sleep(2)

    def _admin_token(self):
        response = self.http.post(f'{self.public_url}/realms/master/protocol/openid-connect/token', timeout=15, data={
            'grant_type': 'password', 'client_id': 'admin-cli', 'username': 'admin', 'password': 'admin'})
        response.raise_for_status()
        return response.json()['access_token']

    def _admin(self, method, path, **kwargs):
        response = self.http.request(method, f'{self.public_url}/admin/realms/{self.realm}{path}', timeout=30,
                                     headers={'Authorization': f'Bearer {self._admin_token()}'}, **kwargs)
        response.raise_for_status()
        return response

    def user_id(self, username):
        return self._admin('GET', '/users', params={'username': username, 'exact': 'true'}).json()[0]['id']

    def logout_user(self, username):
        self._admin('POST', f'/users/{self.user_id(username)}/logout')

    def user_sessions(self, username):
        return self._admin('GET', f'/users/{self.user_id(username)}/sessions').json()

    def update_client(self, **attributes):
        client = self._admin('GET', '/clients', params={'clientId': CLIENT_ID}).json()[0]
        client.setdefault('attributes', {}).update(attributes)
        self._admin('PUT', f"/clients/{client['id']}", json=client)

    def set_frontend_url(self, url):
        realm = self._admin('GET', '').json()
        realm.setdefault('attributes', {})['frontendUrl'] = url or ''
        self._admin('PUT', '', json=realm)

    def password_grant(self, username, password):
        response = self.http.post(f'{self.internal_url}/realms/{self.realm}/protocol/openid-connect/token', timeout=15, data={
            'grant_type': 'password', 'client_id': CLIENT_ID, 'client_secret': 'codeflow-test-secret',
            'username': username, 'password': password, 'scope': 'openid email profile'})
        response.raise_for_status()
        return response.json()

    def reset(self):
        self.update_client(**{'backchannel.logout.url': ''})
        self.set_frontend_url(None)
        self.logout_user('alice')


@pytest.fixture(scope='session')
def keycloak_server():
    from django.conf import settings
    started = False
    if not os.environ.get('OIDC_KC_EXTERNAL'):
        running = _docker('inspect', '-f', '{{.State.Running}}', CONTAINER, check=False)
        if running.returncode != 0 or running.stdout.strip() != 'true':
            _docker('rm', '-f', CONTAINER, check=False)
            port = urlparse(settings.KC_PUBLIC_URL).port or 80
            _docker('run', '-d', '--name', CONTAINER, '-p', f'127.0.0.1:{port}:8080',
                    '--add-host=host.docker.internal:host-gateway',
                    '-e', 'KC_BOOTSTRAP_ADMIN_USERNAME=admin', '-e', 'KC_BOOTSTRAP_ADMIN_PASSWORD=admin',
                    '-v', f'{REALM_FILE}:/opt/keycloak/data/import/codeflow-realm.json:ro',
                    IMAGE, 'start-dev', '--import-realm',
                    f'--hostname={settings.KC_PUBLIC_URL}', '--hostname-backchannel-dynamic=true')
            started = True
    server = Keycloak(settings.KC_PUBLIC_URL, settings.KC_INTERNAL_URL, settings.KC_REALM)
    try:
        server.wait_ready(240)
    except RuntimeError as e:
        logs = _docker('logs', '--tail', '40', CONTAINER, check=False).stdout if started else ''
        raise RuntimeError(f'{e}\n{logs}') from None
    yield server
    if started and not os.environ.get('OIDC_KC_KEEP'):
        _docker('rm', '-f', CONTAINER, check=False)


@pytest.fixture
def kc(keycloak_server):
    from oauth2_authcodeflow.provider import reset_provider
    keycloak_server.reset()
    reset_provider()
    yield keycloak_server
    keycloak_server.reset()
