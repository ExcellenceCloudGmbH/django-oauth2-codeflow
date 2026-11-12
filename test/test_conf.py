from base64 import urlsafe_b64encode
from hashlib import sha1
from types import FunctionType

from django.core.exceptions import ImproperlyConfigured
from pytest import raises

from oauth2_authcodeflow.conf import (
    Settings,
    get_default_django_username,
    import_string_as_func,
    settings,
)
from oauth2_authcodeflow.views import AuthenticateView


def f1() -> str:
    return 'dotted_import'


def test_import_string_as_func():
    with raises(ImportError, match=r"^Could not import 'f1' for API setting 'test'\. ImportError: f1 doesn't look like a module path\.$"):
        import_string_as_func('f1', 'test')
    with raises(ImportError, match=r"^Could not import 'truc\.f1' for API setting 'test'\. ModuleNotFoundError: No module named 'truc'\.$"):
        import_string_as_func('truc.f1', 'test')
    func = import_string_as_func(f'{__name__}.f1', 'test')
    assert func is f1
    assert func() == f1() == 'dotted_import'


def test_get_default_django_username():
    assert get_default_django_username({}) == urlsafe_b64encode(sha1(b'').digest()).decode('ascii').rstrip('=')
    assert get_default_django_username({'email': 'toto@example.com'}) == urlsafe_b64encode(sha1(b'toto@example.com').digest()).decode('ascii').rstrip('=')


def test_Settings_custom_default(settings):
    settings.TOTO = 'titi'
    settings.FOO = True
    settings.BAZ = 'oauth2_authcodeflow.conf.get_default_django_username'
    defaults = {'TOTO': (str, 'toto'), 'FOO': (float, 4.2), 'BAR': (int, ImproperlyConfigured), 'BAZ': (FunctionType, None)}
    mysettings = Settings(defaults)
    assert mysettings.defaults
    assert mysettings.defaults == defaults
    assert mysettings.DEBUG is False
    assert mysettings.TOTO == 'titi'
    with raises(ImproperlyConfigured, match="^Invalid setting: 'FOO' should be of type <class 'float'> and is of type <class 'bool'>$"):
        mysettings.FOO
    with raises(ImproperlyConfigured, match="^Setting 'BAR' not found, it should be defined$"):
        mysettings.BAR
    f = mysettings.BAZ
    assert f is get_default_django_username


def test_Settings_default(settings):
    mysettings = Settings()
    assert mysettings.DEBUG is False
    with raises(ImproperlyConfigured, match="^Setting 'OIDC_RP_CLIENT_SECRET' not found, it should be defined$"):
        mysettings.OIDC_RP_CLIENT_SECRET
    assert mysettings.OIDC_OP_DISCOVERY_DOCUMENT_URL is None
    assert mysettings.OIDC_OP_EXPECTED_EMAIL_CLAIM == 'email'
    assert mysettings.OIDC_RP_SCOPES == ['openid', 'email', 'profile', 'offline_access']
    assert mysettings.OIDC_DJANGO_USERNAME_FUNC is get_default_django_username
    assert mysettings.OIDC_VIEW_AUTHENTICATE is AuthenticateView
    assert mysettings.OIDC_FIRSTNAME_CLAIM == 'given_name'


def test_settings():
    assert isinstance(settings, Settings)


def test_bridge_2_1_defaults(settings):
    """2.1 secure defaults (plan §6.1): no symmetric algorithms, no password scrambling, bounded OP calls."""
    from oauth2_authcodeflow.conf import DEFAULTS
    assert DEFAULTS['OIDC_RP_SIGN_ALGOS_ALLOWED'][1] == ['RS256', 'RS384', 'RS512', 'PS256', 'PS384', 'PS512', 'ES256', 'ES384', 'ES512']
    assert not any(algo.startswith('HS') for algo in DEFAULTS['OIDC_RP_SIGN_ALGOS_ALLOWED'][1])
    assert DEFAULTS['OIDC_UNUSABLE_PASSWORD'][1] is False
    assert DEFAULTS['OIDC_TIMEOUT'][1] == 10
    assert DEFAULTS['OIDC_REFRESH_SKEW_SECONDS'][1] == 30
    assert DEFAULTS['OIDC_RETRY_AFTER_SECONDS'][1] == 5
    assert DEFAULTS['OIDC_REFRESH_LOCK_CACHE_ALIAS'][1] is None
    assert DEFAULTS['OIDC_MIDDLEWARE_NO_REFRESH_URL_PATTERNS'][1] == []
    for name in ('OIDC_RP_SIGN_ALGOS_ALLOWED', 'OIDC_UNUSABLE_PASSWORD', 'OIDC_TIMEOUT'):
        if hasattr(settings, name):
            delattr(settings, name)
    assert settings_value('OIDC_UNUSABLE_PASSWORD') is False
    assert settings_value('OIDC_TIMEOUT') == 10
    # a (connect, read) tuple and a float are valid timeouts
    settings.OIDC_TIMEOUT = (3.05, 8)
    assert settings_value('OIDC_TIMEOUT') == (3.05, 8)
    settings.OIDC_TIMEOUT = 2.5
    assert settings_value('OIDC_TIMEOUT') == 2.5


def settings_value(name):
    from oauth2_authcodeflow.conf import Settings
    return getattr(Settings(), name)
