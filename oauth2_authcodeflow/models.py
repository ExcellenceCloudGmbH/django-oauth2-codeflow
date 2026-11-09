from datetime import (
    datetime,
    timedelta,
    timezone as dt_timezone,
)
from logging import getLogger
from typing import (
    Any,
    Dict,
    Optional,
    cast,
)
from uuid import uuid4

import jwt
from django.conf import settings as dj_settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.db.utils import DatabaseError
from django.utils import timezone
from django.utils.functional import cached_property
from jwt.exceptions import PyJWTError

from .conf import settings

logger = getLogger(__name__)

#: Fields written by a refresh.
TOKEN_FIELDS = [
    'access_token_enc', 'refresh_token_enc', 'id_token_enc', 'access_expires_at',
    'refresh_expires_at', 'sid', 'claims', 'version', 'last_refresh_at',
]
#: Fields written when a sign-in ends.
END_FIELDS = ['ended_at', 'end_reason']


def _cipher():
    from .crypto import get_cipher
    return get_cipher()


class OIDCSession(models.Model):
    """One browser sign-in: the tokens of a Django session, kept out of the session itself.

    Tokens are encrypted at rest (see ``crypto``). Refreshes lock this row
    (``SELECT … FOR UPDATE``), so exactly one refresh runs per sign-in, across processes.
    """
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    user = models.ForeignKey(dj_settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='oidc_sessions')
    user_id: Any  # the user model's primary key type
    django_session_key = models.CharField(max_length=40, blank=True, default='', db_index=True)
    issuer = models.CharField(max_length=512)
    client_id = models.CharField(max_length=255)
    sub = models.CharField(max_length=255)
    # Keycloak session id (``sid`` claim). Not unique: re-logins within one SSO session reuse it.
    sid = models.CharField(max_length=255, blank=True, default='')
    # allow-listed profile claims of the id token / userinfo (never tokens)
    claims: 'models.JSONField[Dict[str, Any]]' = models.JSONField(default=dict, blank=True)
    access_token_enc = models.TextField()
    refresh_token_enc = models.TextField(null=True, blank=True)
    id_token_enc = models.TextField(null=True, blank=True)
    access_expires_at = models.DateTimeField()
    refresh_expires_at = models.DateTimeField(null=True, blank=True)
    # incremented on every token write (permission caches key on it)
    version = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    last_refresh_at = models.DateTimeField(null=True, blank=True)
    last_seen_at = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True, db_index=True)
    end_reason = models.CharField(max_length=32, blank=True, default='')
    migrated_from_legacy = models.BooleanField(default=False)

    class Meta:
        verbose_name = 'OIDC session'
        indexes = [
            models.Index(fields=['issuer', 'sid'], name='oidc_sess_iss_sid_idx'),
            models.Index(fields=['issuer', 'sub'], name='oidc_sess_iss_sub_idx'),
            models.Index(fields=['user', 'ended_at'], name='oidc_sess_user_end_idx'),
            models.Index(fields=['refresh_expires_at'], name='oidc_sess_rexp_idx'),
        ]

    def __str__(self) -> str:
        state = f'ended ({self.end_reason})' if self.ended_at else 'active'
        return f'OIDC session {self.pk} of {self.sub} [{state}]'

    # --- token accessors (decrypted on access only)
    @property
    def access_token(self) -> str:
        return _cipher().decrypt(self.access_token_enc)

    @access_token.setter
    def access_token(self, value: str) -> None:
        self.access_token_enc = _cipher().encrypt(value)

    @property
    def refresh_token(self) -> Optional[str]:
        return _cipher().decrypt(self.refresh_token_enc) if self.refresh_token_enc else None

    @refresh_token.setter
    def refresh_token(self, value: Optional[str]) -> None:
        self.refresh_token_enc = _cipher().encrypt(value) if value else None

    @property
    def id_token(self) -> Optional[str]:
        return _cipher().decrypt(self.id_token_enc) if self.id_token_enc else None

    @id_token.setter
    def id_token(self, value: Optional[str]) -> None:
        self.id_token_enc = _cipher().encrypt(value) if value else None

    # --- state helpers
    @property
    def is_active(self) -> bool:
        return self.ended_at is None

    def remaining(self, now: Optional[datetime] = None) -> float:
        """Seconds of validity left on the access token."""
        return (self.access_expires_at - (now or timezone.now())).total_seconds()

    def mark_ended(self, reason: str, now: Optional[datetime] = None) -> None:
        self.ended_at = now or timezone.now()
        self.end_reason = reason[:32]


class BlacklistedToken(models.Model):
    """Frozen 2.x model. 3.0 never writes or reads it; it is dropped in 4.0."""
    username = models.CharField(max_length=255, editable=False, db_index=True)
    # no max length in RFC6749 but:
    # - https://docs.microsoft.com/en-us/linkedin/shared/authentication/programmatic-refresh-tokens
    # - https://stackoverflow.com/questions/24892496/max-size-for-oauth-token
    # postgres, sqlite, mysql >= 5.0.3 or oracle >= 12c required
    token = models.CharField(max_length=15000, editable=False)
    expires_at = models.DateTimeField(db_index=True)
    blacklisted_at = models.DateTimeField(editable=False, db_index=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['username', 'token'], name='unique_username_token'),
        ]

    @classmethod
    def _get_claims(cls, token: Optional[str]) -> Optional[dict]:
        if not token:
            return None
        try:
            return jwt.decode(token, options={'verify_signature': False})
        except PyJWTError as e:
            logger.debug("cannot decode token claims for blacklist operation: %s", type(e).__name__)
            return None

    @classmethod
    def blacklist(cls, token: Optional[str]) -> Optional['BlacklistedToken']:
        claims = cls._get_claims(token)
        if not claims or not token:
            return None
        username = settings.OIDC_DJANGO_USERNAME_FUNC(claims)
        now = datetime.now(tz=dt_timezone.utc)
        if 'exp' in claims:
            expires_at = datetime.fromtimestamp(claims['exp'], tz=dt_timezone.utc)
        else:
            expires_at = now + timedelta(seconds=settings.OIDC_BLACKLIST_TOKEN_TIMEOUT_SECONDS)
        try:
            return cls.objects.create(username=username, token=token, expires_at=expires_at, blacklisted_at=now)
        except DatabaseError as e:
            logger.warning("cannot blacklist a token: %s", type(e).__name__)
            return None

    @classmethod
    def is_blacklisted(cls, token: Optional[str]) -> bool:
        claims = cls._get_claims(token)
        if not claims or not token:
            return False
        username = settings.OIDC_DJANGO_USERNAME_FUNC(claims)
        return cls.objects.filter(username=username, token=token).exists()

    @classmethod
    def purge(cls) -> int:
        now = datetime.now(tz=dt_timezone.utc)
        nb, _ = cls.objects.filter(expires_at__lte=now).delete()
        return nb

    @cached_property
    def user(self) -> Optional[AbstractUser]:
        User = get_user_model()
        try:
            return cast(AbstractUser, User.objects.get(username=self.username))
        except User.DoesNotExist:
            return None

    def __str__(self) -> str:
        return f"Blacklisted token for {self.username}, expire at {str(self.expires_at)}"
