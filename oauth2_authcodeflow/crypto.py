"""Encryption of the tokens stored on ``OIDCSession`` rows.

``MultiFernet`` over the keys of ``OIDC_TOKEN_ENCRYPTION_KEYS`` (the first one encrypts,
all decrypt). Without explicit keys, keys are derived with HKDF from ``SECRET_KEY`` and
``SECRET_KEY_FALLBACKS``, so a ``SECRET_KEY`` rotation done the Django way keeps sessions
readable; ``manage.py oidc_rotate_keys`` re-encrypts rows with the current key.
"""
import threading
from base64 import urlsafe_b64encode
from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Tuple,
)

from cryptography.fernet import (
    Fernet,
    InvalidToken,
    MultiFernet,
)
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from django.conf import settings as dj_settings
from django.core.exceptions import ImproperlyConfigured
from django.core.signals import setting_changed

from .conf import settings
from .exceptions import OIDCError

_SALT = b'oauth2_authcodeflow.token-encryption.v1'
_INFO = b'oidc-session-tokens'


class TokenDecryptError(OIDCError):
    """A stored token could not be decrypted (keys rotated away, or tampering)."""


def derive_key(secret: Any) -> bytes:
    raw = secret.encode('utf-8') if isinstance(secret, str) else bytes(secret)
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=_SALT, info=_INFO)
    return urlsafe_b64encode(hkdf.derive(raw))


class TokenCipher:
    def __init__(self, keys: Sequence[bytes], derived: bool) -> None:
        if not keys:
            raise ImproperlyConfigured("no token encryption key available")
        self.derived = derived
        self._fernet = MultiFernet([Fernet(k) for k in keys])

    @classmethod
    def from_settings(cls) -> 'TokenCipher':
        secrets: List[Any] = [dj_settings.SECRET_KEY]
        secrets.extend(getattr(dj_settings, 'SECRET_KEY_FALLBACKS', None) or [])
        derived = [derive_key(s) for s in secrets if s]
        explicit = settings.OIDC_TOKEN_ENCRYPTION_KEYS
        if explicit:
            try:
                keys = [k.encode('ascii') if isinstance(k, str) else bytes(k) for k in explicit]
                # The keys derived from SECRET_KEY stay last, for decryption only: moving from
                # the default to an explicit key keeps every sign-in, and oidc_rotate_keys then
                # re-encrypts them with the first explicit key.
                return cls(keys + derived, derived=False)
            except (ValueError, TypeError) as e:
                raise ImproperlyConfigured(f"OIDC_TOKEN_ENCRYPTION_KEYS must hold Fernet keys: {e}")
        return cls(derived, derived=True)

    def encrypt(self, text: str) -> str:
        return self._fernet.encrypt(text.encode('utf-8')).decode('ascii')

    def decrypt(self, token: str) -> str:
        try:
            return self._fernet.decrypt(token.encode('ascii')).decode('utf-8')
        except (InvalidToken, ValueError, AttributeError) as e:
            raise TokenDecryptError('decrypt_failed') from e

    def rotate(self, token: str) -> str:
        try:
            return self._fernet.rotate(token.encode('ascii')).decode('ascii')
        except (InvalidToken, ValueError) as e:
            raise TokenDecryptError('decrypt_failed') from e


_lock = threading.Lock()
_cipher: Optional[Tuple[Any, TokenCipher]] = None


def _fingerprint() -> Any:
    return (
        tuple(settings.OIDC_TOKEN_ENCRYPTION_KEYS or ()),
        dj_settings.SECRET_KEY,
        tuple(getattr(dj_settings, 'SECRET_KEY_FALLBACKS', None) or ()),
    )


def get_cipher() -> TokenCipher:
    global _cipher
    fp = _fingerprint()
    cached = _cipher
    if cached is not None and cached[0] == fp:
        return cached[1]
    with _lock:
        if _cipher is None or _cipher[0] != fp:
            _cipher = (fp, TokenCipher.from_settings())
        return _cipher[1]


def _reset(**kwargs: Any) -> None:
    global _cipher
    if str(kwargs.get('setting', '')) in ('OIDC_TOKEN_ENCRYPTION_KEYS', 'SECRET_KEY', 'SECRET_KEY_FALLBACKS'):
        with _lock:
            _cipher = None


setting_changed.connect(_reset)

__all__ = ['TokenCipher', 'TokenDecryptError', 'derive_key', 'get_cipher']
