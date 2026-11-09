"""System checks (``manage.py check``).

E001 middleware order · E002 HS* algorithm configured · E003 internal discovery URL without
``OIDC_OP_ISSUER`` · E004 the upstream ``django-oauth2-authcodeflow`` is installed too ·
W001 token encryption keys derived from SECRET_KEY · W002 unanchored exempt pattern ·
W003 ``SESSION_COOKIE_AGE`` shorter than 30 days.
"""
from importlib import metadata
from typing import (
    Any,
    List,
    Optional,
)
from urllib.parse import urlparse

from django.conf import settings as dj_settings
from django.core.checks import (
    Error,
    Warning,
    register,
)
from django.utils.module_loading import import_string

from .conf import settings
from .jwt_validation import ASYMMETRIC_ALGORITHMS

ID = 'oauth2_authcodeflow.'
_INTERNAL_SUFFIXES = ('.svc', '.svc.cluster.local', '.cluster.local', '.internal', '.local')


def _cls(path: str) -> Optional[type]:
    try:
        obj = import_string(path)
    except Exception:
        return None
    return obj if isinstance(obj, type) else None


def _index(classes: List[Optional[type]], base: Any) -> Optional[int]:
    for i, cls in enumerate(classes):
        if cls is not None and issubclass(cls, base):
            return i
    return None


@register()
def check_middleware_order(app_configs: Any = None, **kwargs: Any) -> List[Error]:
    from django.contrib.auth.middleware import AuthenticationMiddleware
    from django.contrib.sessions.middleware import SessionMiddleware

    from .middleware import (
        BearerAuthMiddleware,
        LoginRequiredMiddleware,
        OIDCSessionMiddleware,
    )
    classes = [_cls(p) for p in getattr(dj_settings, 'MIDDLEWARE', None) or []]
    oidc = _index(classes, OIDCSessionMiddleware)
    if oidc is None:
        return []
    errors = []
    session = _index(classes, SessionMiddleware)
    authn = _index(classes, AuthenticationMiddleware)
    bearer = _index(classes, BearerAuthMiddleware)
    login = _index(classes, LoginRequiredMiddleware)
    expected = "SessionMiddleware → AuthenticationMiddleware → BearerAuthMiddleware → OIDCSessionMiddleware → LoginRequiredMiddleware"
    if session is None or authn is None or not (session < authn < oidc):
        errors.append(Error("OIDCSessionMiddleware must come after SessionMiddleware and AuthenticationMiddleware", hint=expected, id=ID + 'E001'))
    if bearer is not None and (authn is None or not (authn < bearer < oidc)):
        errors.append(Error("BearerAuthMiddleware must come after AuthenticationMiddleware and before OIDCSessionMiddleware", hint=expected, id=ID + 'E001'))
    if login is not None and login < oidc:
        errors.append(Error("LoginRequiredMiddleware must come after OIDCSessionMiddleware", hint=expected, id=ID + 'E001'))
    return errors


@register()
def check_algorithms(app_configs: Any = None, **kwargs: Any) -> List[Error]:
    algos = list(getattr(dj_settings, 'OIDC_RP_SIGN_ALGOS_ALLOWED', None) or [])
    bad = [a for a in algos if a not in ASYMMETRIC_ALGORITHMS]
    if bad:
        return [Error(
            f"OIDC_RP_SIGN_ALGOS_ALLOWED contains {', '.join(bad)}: only asymmetric algorithms are accepted",
            hint="Anyone holding the client secret could mint HS* tokens. Remove them (they are ignored at runtime).",
            id=ID + 'E002',
        )]
    return []


@register()
def check_internal_issuer(app_configs: Any = None, **kwargs: Any) -> List[Error]:
    url = getattr(dj_settings, 'OIDC_OP_DISCOVERY_DOCUMENT_URL', None)
    if not url or getattr(dj_settings, 'OIDC_OP_ISSUER', None):
        return []
    host = (urlparse(url).hostname or '').lower()
    if host and ('.' not in host or host.endswith(_INTERNAL_SUFFIXES)):
        return [Error(
            "OIDC_OP_DISCOVERY_DOCUMENT_URL points to an internal host but OIDC_OP_ISSUER is not set",
            hint="Set OIDC_OP_ISSUER to the public issuer (e.g. https://auth.example.com/realms/<realm>).",
            id=ID + 'E003',
        )]
    return []


@register()
def check_conflicting_distribution(app_configs: Any = None, **kwargs: Any) -> List[Error]:
    try:
        metadata.distribution('django-oauth2-authcodeflow')
    except metadata.PackageNotFoundError:
        return []
    return [Error(
        "django-oauth2-authcodeflow (upstream) is installed next to django-oauth2-codeflow: both write oauth2_authcodeflow/",
        hint="pip uninstall django-oauth2-authcodeflow && pip install --force-reinstall django-oauth2-codeflow",
        id=ID + 'E004',
    )]


@register()
def check_encryption_keys(app_configs: Any = None, **kwargs: Any) -> List[Warning]:
    if settings.OIDC_TOKEN_ENCRYPTION_KEYS or dj_settings.DEBUG:
        return []
    return [Warning(
        "Token encryption keys are derived from SECRET_KEY",
        hint="Set OIDC_TOKEN_ENCRYPTION_KEYS to dedicated Fernet keys to separate the two secrets.",
        id=ID + 'W001',
    )]


@register()
def check_exempt_patterns(app_configs: Any = None, **kwargs: Any) -> List[Warning]:
    warnings = []
    for name in ('OIDC_LOGIN_REQUIRED_EXEMPT_URL_PATTERNS', 'OIDC_SESSION_EXEMPT_URL_PATTERNS'):
        for pattern in getattr(settings, name) or []:
            if not str(pattern).startswith('^'):
                warnings.append(Warning(
                    f"{name} contains the unanchored pattern {pattern!r}",
                    hint="Unanchored patterns match anywhere in the path (e.g. '/api/x/health…'). Anchor them with '^'.",
                    id=ID + 'W002',
                ))
    return warnings


@register()
def check_session_cookie_age(app_configs: Any = None, **kwargs: Any) -> List[Warning]:
    age = getattr(dj_settings, 'SESSION_COOKIE_AGE', 1209600)
    if age < 30 * 86400:
        return [Warning(
            f"SESSION_COOKIE_AGE is {age} s (< 30 days): Django sessions may end before the OP session",
            hint="Set SESSION_COOKIE_AGE to at least the OP's SSO idle timeout; the middleware slides it.",
            id=ID + 'W003',
        )]
    return []
