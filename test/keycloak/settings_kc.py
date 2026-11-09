"""Settings of the Keycloak integration suite.

Run: ``pytest test/keycloak -m keycloak --ds=test.keycloak.settings_kc``
The app reaches Keycloak through the internal URL; the issuer stays the public URL.
"""
import os
import tempfile

from test.settings import *  # noqa: F401,F403

KEYCLOAK_SUITE = True
KC_PUBLIC_URL = os.environ.get('OIDC_KC_PUBLIC_URL', 'http://localhost:18080')
KC_INTERNAL_URL = os.environ.get('OIDC_KC_INTERNAL_URL', 'http://127.0.0.1:18080')
KC_REALM = 'codeflow'

# a file database: the live server thread gets a connection of its own
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(tempfile.gettempdir(), 'oidc-codeflow-kc.sqlite3'),
        'TEST': {'NAME': os.path.join(tempfile.gettempdir(), 'oidc-codeflow-kc-test.sqlite3')},
    },
}
ALLOWED_HOSTS = ['localhost', '127.0.0.1', 'testserver', 'host.docker.internal']

OIDC_OP_DISCOVERY_DOCUMENT_URL = f'{KC_INTERNAL_URL}/realms/{KC_REALM}/.well-known/openid-configuration'
OIDC_OP_ISSUER = f'{KC_PUBLIC_URL}/realms/{KC_REALM}'
OIDC_RP_CLIENT_ID = 'codeflow-test'
OIDC_RP_CLIENT_SECRET = 'codeflow-test-secret'
STATIC_URL = '/static/'  # the live server's static handler needs it
