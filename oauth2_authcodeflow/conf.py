"""Settings of the library.

Every setting has a default here; projects override them in their Django settings.
Names from 2.x that were renamed are still read (with a ``DeprecationWarning``) and
names that no longer have an effect are reported once. See ``UPGRADING.md``.
"""
import warnings
from base64 import urlsafe_b64encode
from hashlib import sha1
from types import FunctionType
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Set,
    Tuple,
)

from django.conf import settings as dj_settings
from django.core.exceptions import ImproperlyConfigured
from django.core.signals import setting_changed
from django.utils.module_loading import import_string

from . import constants


def import_string_as_func(value: str, attr: str) -> Callable:
    try:
        return import_string(value)
    except ImportError as e:
        raise ImportError(f"Could not import '{value}' for API setting '{attr}'. {e.__class__.__name__}: {e}.")


def get_default_django_username(claims: Dict) -> str:
    """base64 encode of the email hash (sha1)"""
    return urlsafe_b64encode(
        sha1(claims.get('email', '').encode('utf8')).digest()
    ).decode('ascii').rstrip('=')


#: Profile claims kept on ``OIDCSession.claims`` (never tokens). ``'*'`` keeps every claim.
DEFAULT_CLAIMS_ALLOWLIST = [
    'sub', 'iss', 'aud', 'azp', 'sid', 'auth_time', 'acr', 'amr',
    'email', 'email_verified', 'name', 'given_name', 'family_name', 'preferred_username',
    'locale', 'zoneinfo', 'groups', 'roles', 'client_roles', 'client_role', 'realm_role',
]

DEFAULT_CIRCUIT_BREAKER = {
    'failures': 5,          # transient failures within ``window`` seconds open the breaker
    'window': 30,
    'cooldown': 15,         # seconds before a half-open probe is allowed
    'config_cooldown': 300,  # configuration errors keep the breaker open longer
}

DEFAULTS: Dict[str, Tuple[Any, Any]] = {
    # --- views (dotted paths or classes)
    'OIDC_VIEW_AUTHENTICATE': (type, f'{__package__}.views.AuthenticateView'),
    'OIDC_VIEW_CALLBACK': (type, f'{__package__}.views.CallbackView'),
    'OIDC_VIEW_LOGOUT': (type, f'{__package__}.views.LogoutView'),
    'OIDC_VIEW_TOTAL_LOGOUT': (type, f'{__package__}.views.TotalLogoutView'),
    'OIDC_VIEW_BACKCHANNEL_LOGOUT': (type, f'{__package__}.views.BackchannelLogoutView'),
    # --- OpenID provider
    # Discovery document. It may point to an internal URL of the OP (e.g. a cluster
    # Service) as long as the OP resolves back-channel URLs dynamically; the expected
    # public issuer is then given by OIDC_OP_ISSUER.
    'OIDC_OP_DISCOVERY_DOCUMENT_URL': (str, None),
    # Expected ``iss`` of every token. Defaults to the discovery document's ``issuer``.
    # When set, it must equal the discovery ``issuer`` (else: configuration error).
    'OIDC_OP_ISSUER': (str, None),
    # Per-endpoint overrides, applied on top of the discovery document.
    'OIDC_OP_AUTHORIZATION_URL': (str, None),
    'OIDC_OP_TOKEN_URL': (str, None),
    'OIDC_OP_USERINFO_URL': (str, None),
    'OIDC_OP_JWKS_URL': (str, None),
    'OIDC_OP_END_SESSION_URL': (str, None),
    'OIDC_OP_REVOCATION_URL': (str, None),
    # Complete the id token claims with the userinfo endpoint at login.
    'OIDC_OP_FETCH_USER_INFO': (bool, True),
    # Total logout redirects to the OP end-session endpoint.
    'OIDC_OP_TOTAL_LOGOUT': (bool, True),
    'OIDC_OP_EXPECTED_EMAIL_CLAIM': (str, 'email'),
    'OIDC_OP_EXPECTED_CLAIMS': (list, []),
    # --- relying party (client)
    'OIDC_RP_CLIENT_ID': (str, ImproperlyConfigured),
    'OIDC_RP_CLIENT_SECRET': (str, ImproperlyConfigured),
    # 'client_secret_post', 'client_secret_basic' or 'none'.
    # None: 'client_secret_post' when a secret is configured, else 'none'.
    'OIDC_RP_CLIENT_AUTH_METHOD': (str, None),
    'OIDC_RP_USE_PKCE': (bool, True),
    'OIDC_RP_FORCE_CONSENT_PROMPT': (bool, False),
    'OIDC_RP_AZURE_SPA': (bool, False),
    'OIDC_RP_SCOPES': (list, ['openid', 'email', 'profile']),
    'OIDC_RP_USERINFO_CLAIMS': (dict, None),
    'OIDC_RP_TOKEN_CLAIMS': (dict, None),
    # Asymmetric algorithms only; HS* is never accepted (system check E002).
    'OIDC_RP_SIGN_ALGOS_ALLOWED': (list, ['RS256', 'PS256', 'ES256']),
    # Static public key (PEM key or certificate) that overrides the JWKS.
    'OIDC_RP_IDP_SIGN_KEY': ((bytes, str), None),
    'OIDC_ID_TOKEN_LEEWAY_SECONDS': (int, 30),
    # --- login flow
    'OIDC_LOGIN_STATE_TTL_SECONDS': (int, 600),
    'OIDC_LOGIN_MAX_PENDING': (int, 5),
    # Extra hosts allowed in ``next``/``fail`` (the request host is always allowed).
    'OIDC_ALLOWED_REDIRECT_HOSTS': (list, []),
    'OIDC_RANDOM_SIZE': (int, 32),
    'OIDC_REDIRECT_OK_FIELD_NAME': (str, 'next'),
    'OIDC_REDIRECT_ERROR_FIELD_NAME': (str, 'fail'),
    'OIDC_LOGIN_URL': (str, None),
    # --- users
    'OIDC_CREATE_USER': (bool, True),
    'OIDC_DJANGO_USERNAME_FUNC': (FunctionType, get_default_django_username),
    'OIDC_EMAIL_CLAIM': ((str, FunctionType), None),
    'OIDC_FIRSTNAME_CLAIM': ((str, FunctionType), 'given_name'),
    'OIDC_LASTNAME_CLAIM': ((str, FunctionType), 'family_name'),
    'OIDC_EXTEND_USER': (FunctionType, None),
    # True scrambles the password at every login, which ends the user's other Django
    # sessions (session auth hash). Only applied on creation by default.
    'OIDC_UNUSABLE_PASSWORD': (bool, False),
    'OIDC_CLAIMS_ALLOWLIST': (list, DEFAULT_CLAIMS_ALLOWLIST),
    # --- HTTP to the OP
    'OIDC_HTTP_TIMEOUT': ((tuple, list, int, float), (3.05, 8)),
    'OIDC_CA_BUNDLE': ((bool, str), True),
    'OIDC_PROXY': (dict, None),
    'OIDC_HTTP_POOL_MAXSIZE': (int, 20),
    'OIDC_DISCOVERY_CACHE_SECONDS': (int, 3600),
    'OIDC_JWKS_CACHE_SECONDS': (int, 3600),
    'OIDC_JWKS_MIN_REFETCH_SECONDS': (int, 30),
    # --- sessions
    'OIDC_SESSION_KEY': (str, '_oidc_session_id'),
    # 'eager': refresh in the middleware; 'lazy': only in get_access_token().
    'OIDC_REFRESH_MODE': (str, 'eager'),
    'OIDC_REFRESH_SKEW_SECONDS': (int, 60),
    'OIDC_REFRESH_WAIT_SECONDS': ((int, float), 10),
    'OIDC_ACCESS_MIN_USABLE_SECONDS': (int, 5),
    'OIDC_CIRCUIT_BREAKER': (dict, DEFAULT_CIRCUIT_BREAKER),
    'OIDC_RETRY_AFTER_SECONDS': (int, 5),
    'OIDC_SESSION_RENEW_INTERVAL_SECONDS': (int, 3600),
    # Absolute cap of a sign-in. None: the OP decides (refresh fails with invalid_grant).
    'OIDC_SESSION_MAX_AGE_SECONDS': (int, None),
    # Fernet keys (str or bytes); the first encrypts, all decrypt. None: derived from
    # SECRET_KEY and SECRET_KEY_FALLBACKS with HKDF.
    'OIDC_TOKEN_ENCRYPTION_KEYS': (list, None),
    'OIDC_LEGACY_SESSION_MIGRATION': (bool, True),
    # --- middlewares
    # Paths without session/refresh handling (health checks, static assets, ...).
    'OIDC_SESSION_EXEMPT_URL_PATTERNS': (list, []),
    # Paths that LoginRequiredMiddleware lets through anonymous requests.
    'OIDC_LOGIN_REQUIRED_EXEMPT_URL_PATTERNS': (list, []),
    # Paths that always get JSON answers (never a redirect).
    'OIDC_API_URL_PATTERNS': (list, ['^/api/']),
    # Navigation requests without a session are redirected to the login view.
    'OIDC_NAVIGATION_REDIRECT': (bool, True),
    # --- bearer tokens
    'OIDC_AUTHORIZATION_HEADER_PREFIX': (str, 'Bearer'),
    'OIDC_BEARER_AUDIENCES': (list, None),
    'OIDC_BEARER_ACCEPT_AZP': (bool, False),
    'OIDC_BEARER_TOKEN_TYPE': (str, 'Bearer'),
    'OIDC_BEARER_LEEWAY_SECONDS': (int, 30),
    # --- logout
    # RFC 7009 revocation of the refresh token on (total) logout. Off by default: a
    # Keycloak client session is shared by every application using the same client.
    'OIDC_REVOKE_ON_LOGOUT': (bool, False),
    'OIDC_BACKCHANNEL_LOGOUT_ENABLED': (bool, True),
    'OIDC_BACKCHANNEL_LOGOUT_MAX_AGE_SECONDS': (int, 300),
    'OIDC_CACHE_ALIAS': (str, 'default'),
    # 'auto' (when channels is installed), 'on' or 'off'.
    'OIDC_CHANNELS_NOTIFY': (str, 'auto'),
    # --- frozen 2.x model
    'OIDC_BLACKLIST_TOKEN_TIMEOUT_SECONDS': (int, 7 * 86400),
}


def _force_secret_to_auth_method(value: Any) -> Optional[str]:
    return 'client_secret_post' if value else None


#: 2.x names that are still read: old name -> (new name, value transformation)
DEPRECATED: Dict[str, Tuple[str, Optional[Callable[[Any], Any]]]] = {
    'OIDC_TIMEOUT': ('OIDC_HTTP_TIMEOUT', None),
    'OIDC_RP_FORCE_SECRET_WITH_PKCE': ('OIDC_RP_CLIENT_AUTH_METHOD', _force_secret_to_auth_method),
    'OIDC_MIDDLEWARE_NO_AUTH_URL_PATTERNS': ('OIDC_LOGIN_REQUIRED_EXEMPT_URL_PATTERNS', None),
    'OIDC_MIDDLEWARE_NO_REFRESH_URL_PATTERNS': ('OIDC_SESSION_EXEMPT_URL_PATTERNS', None),
    'OIDC_MIDDLEWARE_API_URL_PATTERNS': ('OIDC_API_URL_PATTERNS', None),
    'OIDC_MIDDLEWARE_LOGIN_REQUIRED_REDIRECT': ('OIDC_NAVIGATION_REDIRECT', None),
}

#: 2.x names without effect in 3.0: name -> explanation
REMOVED: Dict[str, str] = {
    'OIDC_MIDDLEWARE_SESSION_TIMEOUT_SECONDS': (
        "the OP decides how long a sign-in lasts; use OIDC_SESSION_MAX_AGE_SECONDS for an absolute cap"
    ),
    'OIDC_VIEW_LOGOUT_BY_OP': "the front-channel logout view was removed; use the back-channel logout endpoint",
}

_NEW_TO_OLD = {new: old for old, (new, _) in DEPRECATED.items()}


class Settings:
    """A settings object that allows settings to be accessed as properties."""
    defaults: Dict[str, Tuple[Any, Any]]

    def __init__(self, defaults: Optional[Dict[str, Tuple[Any, Any]]] = None) -> None:
        self.defaults = defaults or DEFAULTS
        self._cache: Set[str] = set()
        self._warned: Set[str] = set()
        setting_changed.connect(self.reload)

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith('_'):
            raise AttributeError(attr)
        atype, def_val = self.defaults.get(attr, (None, None))
        val = self._raw(attr, def_val)
        if atype is None:  # other setting
            atype = type(val),
        elif not isinstance(atype, tuple):
            atype = atype,
        val = self._check_type_and_get_value(attr, val, atype)
        # cache the result
        self._cache.add(attr)
        setattr(self, attr, val)
        return val

    def _raw(self, attr: str, default: Any) -> Any:
        if hasattr(dj_settings, attr):
            return getattr(dj_settings, attr)
        old = _NEW_TO_OLD.get(attr)
        if old and hasattr(dj_settings, old):
            _, transform = DEPRECATED[old]
            self._warn_once(old, f"{old} is deprecated, use {attr} instead")
            value = getattr(dj_settings, old)
            return transform(value) if transform else value
        return default

    def _warn_once(self, name: str, message: str) -> None:
        if name in self._warned:
            return
        self._warned.add(name)
        warnings.warn(message, DeprecationWarning, stacklevel=4)

    def warn_removed(self) -> None:
        """Warn once about 2.x settings that have no effect anymore (called at app ready)."""
        for name, why in REMOVED.items():
            if hasattr(dj_settings, name):
                self._warn_once(name, f"{name} has no effect anymore: {why}")

    def _check_type_and_get_value(self, attr: str, val: Any, atype: tuple) -> Any:
        if (FunctionType in atype or type in atype) and isinstance(val, str) and '.' in val:
            val = import_string_as_func(val, attr)  # dotted strings to function
        if val is ImproperlyConfigured:
            raise ImproperlyConfigured(f"Setting '{attr}' not found, it should be defined")
        if val is not None and not isinstance(val, atype):
            raise ImproperlyConfigured(f"Invalid setting: '{attr}' should be of type {', '.join(map(str, atype))} and is of type {type(val)}")
        return val

    def reload(self, *args, **kwargs) -> None:
        attr = str(kwargs.get('setting'))
        names = {attr}
        if attr in DEPRECATED:
            names.add(DEPRECATED[attr][0])
        for name in names:
            # hasattr cannot be used because it relies on getattr and __getattr__, so use the _cache
            if name in self._cache:
                self._cache.remove(name)
            try:
                delattr(self, name)
            except AttributeError:
                pass


settings = Settings()


__all__ = [
    'DEFAULTS',
    'DEPRECATED',
    'REMOVED',
    'Settings',
    'constants',
    'get_default_django_username',
    'import_string_as_func',
    'settings',
]
