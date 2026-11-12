"""``request.oidc``: what the current request knows about its sign-in.

Set by ``OIDCSessionMiddleware`` (session sign-ins) and ``BearerAuthMiddleware`` (bearer
tokens); ``None`` for anonymous requests and non-OIDC logins. Tokens are not attributes:
ask ``tokens.get_access_token(request)``, which refreshes when needed.
"""
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
)

from .conf import settings
from .jwt_validation import unverified_claims

if TYPE_CHECKING:  # pragma: no cover
    from .models import OIDCSession


class OIDCContext:
    def __init__(
        self,
        *,
        session_id: Optional[str],
        sub: str,
        sid: str,
        issuer: str,
        claims: Dict[str, Any],
        access_expires_at: Optional[datetime],
        refresh_expires_at: Optional[datetime],
        version: int,
        is_bearer: bool,
        row: Optional['OIDCSession'] = None,
        bearer_token: Optional[str] = None,
        access_claims: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.session_id = session_id
        self.sub = sub
        self.sid = sid
        self.issuer = issuer
        self.claims = claims
        self.access_expires_at = access_expires_at
        self.refresh_expires_at = refresh_expires_at
        self.version = version
        self.is_bearer = is_bearer
        self._row = row
        self._bearer_token = bearer_token
        self._access_claims = access_claims

    @classmethod
    def from_row(cls, row: 'OIDCSession') -> 'OIDCContext':
        return cls(
            session_id=str(row.pk),
            sub=row.sub,
            sid=row.sid,
            issuer=row.issuer,
            claims=dict(row.claims or {}),
            access_expires_at=row.access_expires_at,
            refresh_expires_at=row.refresh_expires_at,
            version=row.version,
            is_bearer=False,
            row=row,
        )

    @classmethod
    def from_bearer(cls, token: str, claims: Dict[str, Any]) -> 'OIDCContext':
        from datetime import timezone as dt_timezone
        exp = claims.get('exp')
        expires = datetime.fromtimestamp(exp, tz=dt_timezone.utc) if isinstance(exp, (int, float)) else None
        return cls(
            session_id=None,
            sub=str(claims.get('sub', '')),
            sid=str(claims.get('sid', '') or ''),
            issuer=str(claims.get('iss', '')),
            claims=dict(claims),
            access_expires_at=expires,
            refresh_expires_at=None,
            version=0,
            is_bearer=True,
            bearer_token=token,
            access_claims=dict(claims),
        )

    def _update(self, row: 'OIDCSession') -> None:
        self._row = row
        self.claims = dict(row.claims or {})
        self.sid = row.sid
        self.access_expires_at = row.access_expires_at
        self.refresh_expires_at = row.refresh_expires_at
        if row.version != self.version:
            self._access_claims = None
        self.version = row.version

    @property
    def access_claims(self) -> Dict[str, Any]:
        """Claims of the current access token (read without signature check: it came from the OP)."""
        if self._access_claims is None:
            token = self._bearer_token
            if token is None and self._row is not None:
                try:
                    token = self._row.access_token
                except Exception:
                    token = None
            self._access_claims = unverified_claims(token)
        return self._access_claims

    def client_roles(self, client_id: Optional[str] = None) -> List[str]:
        """Roles of the client in the access token (``resource_access``), else the ``client_roles`` claim."""
        cid = client_id or settings.OIDC_RP_CLIENT_ID
        roles: List[str] = []
        access = (self.access_claims.get('resource_access') or {}).get(cid) or {}
        for role in access.get('roles') or []:
            if isinstance(role, str) and role not in roles:
                roles.append(role)
        if roles:
            return roles
        raw = self.claims.get('client_roles', self.access_claims.get('client_roles'))
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, (list, tuple)):
            return [r for r in raw if isinstance(r, str)]
        if isinstance(raw, dict):
            values = raw.get(cid, [])
            return [r for r in (values if isinstance(values, list) else [values]) if isinstance(r, str)]
        return []

    def __repr__(self) -> str:
        kind = 'bearer' if self.is_bearer else f'session {self.session_id}'
        return f'<OIDCContext {kind} sub={self.sub}>'


__all__ = ['OIDCContext']
