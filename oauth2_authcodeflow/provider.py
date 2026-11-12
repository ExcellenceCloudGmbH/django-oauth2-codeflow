"""OpenID provider metadata and signing keys, cached per process.

- The discovery document is cached for ``OIDC_DISCOVERY_CACHE_SECONDS`` and kept (stale)
  when a refresh fails, so a short OP outage does not break sign-in validation.
- ``OIDC_OP_ISSUER`` (the public issuer) must equal the discovery ``issuer``; this is what
  makes an internal discovery URL safe (the OP resolves back-channel URLs dynamically while
  tokens keep the public issuer).
- Explicit ``OIDC_OP_*_URL`` settings override discovered endpoints.
- JWKS are cached for ``OIDC_JWKS_CACHE_SECONDS``; an unknown ``kid`` triggers at most one
  refetch every ``OIDC_JWKS_MIN_REFETCH_SECONDS`` (key rotation without a stampede).
"""
import logging
import threading
from dataclasses import dataclass
from time import monotonic
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import jwt
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.x509 import load_pem_x509_certificate
from django.core.signals import setting_changed
from requests import RequestException

from . import events
from .conf import settings
from .exceptions import (
    OIDCAuthUnavailable,
    OIDCConfigurationError,
    OIDCTokenError,
)
from .http import get_client

logger = logging.getLogger(__name__)

_KTY_FOR_ALG = {'RS': 'RSA', 'PS': 'RSA', 'ES': 'EC', 'Ed': 'OKP'}


@dataclass(frozen=True)
class ProviderMetadata:
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: Optional[str] = None
    jwks_uri: Optional[str] = None
    end_session_endpoint: Optional[str] = None
    revocation_endpoint: Optional[str] = None
    claims_parameter_supported: bool = False
    discovered_issuer: Optional[str] = None


@dataclass(frozen=True)
class SigningKey:
    kid: Optional[str]
    kty: str
    alg: Optional[str]
    key: Any


def _kty_for(alg: str) -> Optional[str]:
    return _KTY_FOR_ALG.get(alg[:2])


def _load_static_key(value: Any) -> Any:
    data = value.encode('ascii') if isinstance(value, str) else value
    if data.lstrip().startswith(b'-----BEGIN CERTIFICATE-----'):
        return load_pem_x509_certificate(data).public_key()
    return load_pem_public_key(data)


class Provider:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._metadata: Optional[ProviderMetadata] = None
        self._metadata_fetched = 0.0
        self._keys: List[SigningKey] = []
        self._keys_fetched = 0.0
        self._last_refetch = -1e9
        self._static: Optional[Tuple[Any, Any]] = None

    # --- metadata
    def metadata(self) -> ProviderMetadata:
        ttl = settings.OIDC_DISCOVERY_CACHE_SECONDS
        md = self._metadata
        if md is not None and monotonic() - self._metadata_fetched < ttl:
            return md
        with self._lock:
            now = monotonic()
            if self._metadata is not None and now - self._metadata_fetched < ttl:
                return self._metadata
            try:
                md = self._load_metadata()
            except OIDCAuthUnavailable:
                if self._metadata is not None:
                    events.emit('oidc.discovery.stale', logging.WARNING)
                    # retry within a minute, keep serving the last good document meanwhile
                    self._metadata_fetched = now - ttl + min(60, ttl)
                    return self._metadata
                raise
            self._metadata = md
            self._metadata_fetched = now
            return md

    def _load_metadata(self) -> ProviderMetadata:
        doc: Dict[str, Any] = {}
        url = settings.OIDC_OP_DISCOVERY_DOCUMENT_URL
        if url:
            try:
                resp = get_client().get(url, headers={'Accept': 'application/json'})
            except RequestException as e:
                events.emit('oidc.discovery.failed', logging.WARNING, error=type(e).__name__)
                raise OIDCAuthUnavailable('discovery_unreachable')
            if resp.status_code != 200:
                events.emit('oidc.discovery.failed', logging.WARNING, status=resp.status_code)
                raise OIDCAuthUnavailable('discovery_unavailable')
            try:
                doc = resp.json()
            except ValueError:
                raise OIDCAuthUnavailable('discovery_invalid')
            if not isinstance(doc, dict):
                raise OIDCAuthUnavailable('discovery_invalid')
        discovered = doc.get('issuer')
        configured = settings.OIDC_OP_ISSUER
        if configured and discovered and configured.rstrip('/') != str(discovered).rstrip('/'):
            events.emit('oidc.config.issuer_mismatch', logging.ERROR, configured=configured, discovered=discovered)
            raise OIDCConfigurationError("OIDC_OP_ISSUER does not match the discovery document's issuer")

        def pick(setting: str, key: str) -> Optional[str]:
            return getattr(settings, setting) or doc.get(key)

        md = ProviderMetadata(
            issuer=configured or discovered or '',
            authorization_endpoint=pick('OIDC_OP_AUTHORIZATION_URL', 'authorization_endpoint') or '',
            token_endpoint=pick('OIDC_OP_TOKEN_URL', 'token_endpoint') or '',
            userinfo_endpoint=pick('OIDC_OP_USERINFO_URL', 'userinfo_endpoint'),
            jwks_uri=pick('OIDC_OP_JWKS_URL', 'jwks_uri'),
            end_session_endpoint=pick('OIDC_OP_END_SESSION_URL', 'end_session_endpoint'),
            revocation_endpoint=pick('OIDC_OP_REVOCATION_URL', 'revocation_endpoint'),
            claims_parameter_supported=bool(doc.get('claims_parameter_supported', False)) if url else True,
            discovered_issuer=discovered,
        )
        if not md.authorization_endpoint or not md.token_endpoint:
            raise OIDCConfigurationError("the authorization and token endpoints are unknown (discovery or OIDC_OP_*_URL)")
        return md

    def expected_issuer(self) -> Optional[str]:
        return self.metadata().issuer or None

    # --- keys
    def key_for(self, kid: Optional[str], alg: str) -> Any:
        """The public key for a token header, refetching the JWKS once on an unknown kid."""
        static = settings.OIDC_RP_IDP_SIGN_KEY
        if static:
            cached = self._static
            if cached is None or cached[0] is not static:
                cached = (static, _load_static_key(static))
                self._static = cached
            return cached[1]
        key = self._match(self._jwks(), kid, alg)
        if key is None and self._may_refetch():
            key = self._match(self._jwks(force=True), kid, alg)
        if key is None:
            raise OIDCTokenError('unknown_signing_key')
        return key

    def _may_refetch(self) -> bool:
        with self._lock:
            now = monotonic()
            if now - self._last_refetch < settings.OIDC_JWKS_MIN_REFETCH_SECONDS:
                return False
            self._last_refetch = now
            return True

    @staticmethod
    def _match(keys: List[SigningKey], kid: Optional[str], alg: str) -> Any:
        kty = _kty_for(alg)
        for candidate in keys:
            if kid is not None and candidate.kid != kid:
                continue
            if kty and candidate.kty != kty:
                continue
            if candidate.alg and candidate.alg != alg:
                continue
            return candidate.key
        return None

    def _jwks(self, force: bool = False) -> List[SigningKey]:
        ttl = settings.OIDC_JWKS_CACHE_SECONDS
        if not force and self._keys and monotonic() - self._keys_fetched < ttl:
            return self._keys
        with self._lock:
            if not force and self._keys and monotonic() - self._keys_fetched < ttl:
                return self._keys
            uri = self.metadata().jwks_uri
            if not uri:
                raise OIDCConfigurationError("no jwks_uri (discovery or OIDC_OP_JWKS_URL) and no OIDC_RP_IDP_SIGN_KEY")
            try:
                resp = get_client().get(uri, headers={'Accept': 'application/json'})
                if resp.status_code != 200:
                    raise OIDCAuthUnavailable('jwks_unavailable')
                data = resp.json()
            except (RequestException, ValueError):
                if self._keys:
                    return self._keys
                raise OIDCAuthUnavailable('jwks_unavailable')
            except OIDCAuthUnavailable:
                if self._keys:
                    return self._keys
                raise
            self._keys = self._parse(data)
            self._keys_fetched = monotonic()
            return self._keys

    @staticmethod
    def _parse(data: Any) -> List[SigningKey]:
        keys: List[SigningKey] = []
        for jwk in (data or {}).get('keys', []) if isinstance(data, dict) else []:
            if not isinstance(jwk, dict) or jwk.get('use') == 'enc':
                continue
            try:
                parsed = jwt.PyJWK(jwk)
            except Exception:  # unsupported key type or algorithm
                logger.debug("skipping an unsupported JWK (kid=%s)", jwk.get('kid'))
                continue
            keys.append(SigningKey(kid=jwk.get('kid'), kty=str(jwk.get('kty')), alg=jwk.get('alg'), key=parsed.key))
        return keys


_provider_lock = threading.Lock()
_provider: Optional[Provider] = None


def get_provider() -> Provider:
    global _provider
    provider = _provider
    if provider is None:
        with _provider_lock:
            if _provider is None:
                _provider = Provider()
            provider = _provider
    return provider


def set_provider(provider: Optional[Provider]) -> None:
    """Replace the process provider (tests, ``testing.FakeOP``)."""
    global _provider
    with _provider_lock:
        _provider = provider


def reset_provider() -> None:
    set_provider(None)


def _reset(**kwargs: Any) -> None:
    setting = str(kwargs.get('setting', ''))
    if setting.startswith(('OIDC_OP_', 'OIDC_RP_IDP_SIGN_KEY', 'OIDC_DISCOVERY', 'OIDC_JWKS')) and type(_provider) is Provider:
        reset_provider()


setting_changed.connect(_reset)

__all__ = ['Provider', 'ProviderMetadata', 'SigningKey', 'get_provider', 'reset_provider', 'set_provider']
