"""The sign-in service: loading, refreshing and ending ``OIDCSession`` rows.

Refresh is single-flight across threads *and* processes:

1. a fast path when the access token is valid for more than ``OIDC_REFRESH_SKEW_SECONDS``;
2. an in-process keyed lock (requests whose token is still usable never wait on it);
3. ``SELECT … FOR UPDATE`` on the row (``NOWAIT`` while the token is usable), with a
   transaction-scoped Postgres ``lock_timeout``; the row is re-read after the lock, so a
   request that waited sees the tokens another process just wrote;
4. the token-endpoint call, bounded by HTTP timeouts and the circuit breaker.

The outcome is computed inside the transaction and raised after it, so a sign-in that
``invalid_grant`` ended stays ended. A refresh never runs inside a caller's atomic block.
Only ``invalid_grant`` ends a sign-in; transient and configuration errors keep it and serve
the still-valid token, or answer ``OIDCAuthUnavailable`` (503).
"""
import logging
import time
from datetime import (
    datetime,
    timedelta,
    timezone as dt_timezone,
)
from functools import partial
from importlib import import_module
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)

from django.conf import settings as dj_settings
from django.contrib import auth
from django.contrib.auth import BACKEND_SESSION_KEY
from django.core.exceptions import ValidationError
from django.db import (
    DatabaseError,
    connections,
    router,
    transaction,
)
from django.http import HttpRequest
from django.utils import timezone
from django.utils.module_loading import import_string

from . import (
    constants,
    events,
    signals,
    token_endpoint,
)
from .breaker import get_breaker
from .conf import settings
from .crypto import TokenDecryptError
from .exceptions import (
    OIDCAuthUnavailable,
    OIDCConfigurationError,
    OIDCSessionEnded,
    OIDCTokenError,
)
from .jwt_validation import (
    unverified_claims,
    validate_id_token,
)
from .locks import keyed_lock
from .models import (
    END_FIELDS,
    TOKEN_FIELDS,
    OIDCSession,
)

logger = logging.getLogger(__name__)

#: Postgres SQLSTATEs that mean "someone else holds the row": lock_not_available,
#: serialization_failure, deadlock_detected.
LOCK_CONTENTION_CODES = frozenset({'55P03', '40001', '40P01'})
_DEFAULT_ACCESS_LIFETIME = 300


# --- helpers ---------------------------------------------------------------------------
def _now() -> datetime:
    return timezone.now()


def _from_ts(ts: float) -> datetime:
    value = datetime.fromtimestamp(float(ts), tz=dt_timezone.utc)
    if not dj_settings.USE_TZ:
        value = timezone.make_naive(value, timezone.get_default_timezone())
    return value


def _token_exp(token: Optional[str]) -> Optional[datetime]:
    exp = unverified_claims(token).get('exp')
    return _from_ts(exp) if isinstance(exp, (int, float)) else None


def allowlisted(claims: Dict[str, Any]) -> Dict[str, Any]:
    """The profile claims kept on a row (``OIDC_CLAIMS_ALLOWLIST``)."""
    allow = settings.OIDC_CLAIMS_ALLOWLIST or []
    if '*' in allow:
        return dict(claims)
    return {key: value for key, value in claims.items() if key in allow}


def _refresh_expiry(data: Dict[str, Any], refresh: Optional[str], now: datetime) -> Optional[datetime]:
    raw = data.get('refresh_expires_in')
    if raw is not None:
        try:
            seconds = int(raw)
        except (TypeError, ValueError):
            seconds = None
        if seconds == 0:  # Keycloak: no expiry (offline token)
            return None
        if seconds:
            return now + timedelta(seconds=seconds)
    return _token_exp(refresh) if refresh else None


def _access_expiry(data: Dict[str, Any], access: str, now: datetime) -> datetime:
    expires_in = data.get('expires_in')
    try:
        if expires_in:
            return now + timedelta(seconds=int(expires_in))
    except (TypeError, ValueError):
        pass
    return _token_exp(access) or now + timedelta(seconds=_DEFAULT_ACCESS_LIFETIME)


def _db() -> str:
    return router.db_for_write(OIDCSession) or 'default'


def _in_caller_atomic(alias: str) -> bool:
    connection = connections[alias]
    blocks = getattr(connection, 'atomic_blocks', None)
    if blocks is None:
        return connection.in_atomic_block
    # Django's TestCase wraps each test in atomic blocks flagged _from_testcase: not a caller's transaction
    return any(not getattr(block, '_from_testcase', False) for block in blocks)


def _is_contention(error: BaseException) -> bool:
    for candidate in (error, getattr(error, '__cause__', None), getattr(error, '__context__', None)):
        code = getattr(candidate, 'pgcode', None) or getattr(candidate, 'sqlstate', None)
        if code in LOCK_CONTENTION_CODES:
            return True
    return False


def _retry_after() -> int:
    breaker = get_breaker()
    return breaker.retry_after() if breaker.is_open() else int(settings.OIDC_RETRY_AFTER_SECONDS)


def _reload(pk: Any) -> OIDCSession:
    """The current row, or OIDCSessionEnded when it is gone or ended."""
    row = OIDCSession.objects.filter(pk=pk).first()
    if row is None:
        raise OIDCSessionEnded('missing_row')
    if row.ended_at is not None:
        raise OIDCSessionEnded(row.end_reason or 'ended')
    return row


def _usable_or_unavailable(row: OIDCSession, category: str) -> OIDCSession:
    if row.remaining(_now()) > settings.OIDC_ACCESS_MIN_USABLE_SECONDS:
        return row
    raise OIDCAuthUnavailable(category or 'refresh_failed', retry_after=_retry_after())


# --- creation -------------------------------------------------------------------------
def create_session(
    *,
    user: Any,
    token_data: Dict[str, Any],
    id_claims: Dict[str, Any],
    extra_claims: Optional[Dict[str, Any]] = None,
    django_session_key: str = '',
) -> OIDCSession:
    """A new row from a token-endpoint response (code exchange)."""
    now = _now()
    access = token_data['access_token']
    refresh = token_data.get('refresh_token')
    claims = dict(id_claims)
    claims.update(extra_claims or {})
    row = OIDCSession(
        user=user,
        django_session_key=django_session_key or '',
        issuer=str(id_claims.get('iss', '')),
        client_id=settings.OIDC_RP_CLIENT_ID,
        sub=str(id_claims.get('sub', '')),
        sid=str(id_claims.get('sid', '') or ''),
        claims=allowlisted(claims),
        access_expires_at=_access_expiry(token_data, access, now),
        refresh_expires_at=_refresh_expiry(token_data, refresh, now),
    )
    row.access_token = access
    row.refresh_token = refresh
    row.id_token = token_data.get('id_token')
    row.save()
    return row


def attach_access_token(request: HttpRequest, user: Any, token: str, claims: Dict[str, Any]) -> OIDCSession:
    """A refresh-less row for a request authenticated by an already validated access token
    (embeds). It ends when the access token expires."""
    exp = claims.get('exp')
    row = OIDCSession(
        user=user,
        django_session_key=getattr(request.session, 'session_key', None) or '',
        issuer=str(claims.get('iss', '')),
        client_id=settings.OIDC_RP_CLIENT_ID,
        sub=str(claims.get('sub', '')),
        sid=str(claims.get('sid', '') or ''),
        claims=allowlisted(claims),
        access_expires_at=_from_ts(exp) if isinstance(exp, (int, float)) else _now() + timedelta(seconds=_DEFAULT_ACCESS_LIFETIME),
    )
    row.access_token = token
    row.save()
    request.session[settings.OIDC_SESSION_KEY] = str(row.pk)
    request._oidc_row = row  # type: ignore[attr-defined]
    events.emit('oidc.session.started', logging.INFO, session_id=str(row.pk), user_id=row.user_id, kind='attached')
    return row


# --- loading --------------------------------------------------------------------------
def _authenticated_via_oidc(request: HttpRequest) -> bool:
    path = request.session.get(BACKEND_SESSION_KEY)
    if not path:
        return False
    try:
        backend = import_string(path)
    except ImportError:
        return False
    from .auth import AuthenticationBackend
    return isinstance(backend, type) and issubclass(backend, AuthenticationBackend)


def has_legacy_keys(session: Any) -> bool:
    return constants.SESSION_ACCESS_TOKEN in session


def load_for_request(request: HttpRequest) -> Optional[OIDCSession]:
    """The active row of this request's Django session.

    None for anonymous requests and for logins of other backends (they pass through
    untouched). Raises OIDCSessionEnded when an OIDC sign-in exists but is over.
    """
    session = getattr(request, 'session', None)
    if session is None:
        return None
    key = session.get(settings.OIDC_SESSION_KEY)
    if not key:
        if not _authenticated_via_oidc(request):
            return None
        if settings.OIDC_LEGACY_SESSION_MIGRATION and has_legacy_keys(session):
            row = migrate_legacy(request)
            if row is not None:
                return _check_row(request, row)
        raise OIDCSessionEnded('missing_oidc_session')
    try:
        row = OIDCSession.objects.filter(pk=key).first()
    except (ValidationError, ValueError):
        row = None
    if row is None:
        raise OIDCSessionEnded('missing_row')
    return _check_row(request, row)


def _check_row(request: HttpRequest, row: OIDCSession) -> OIDCSession:
    if row.ended_at is not None:
        raise OIDCSessionEnded(row.end_reason or 'ended')
    user = getattr(request, 'user', None)
    if user is None or not user.is_authenticated or user.pk != row.user_id:
        end_session(row, 'user_mismatch', delete_django_session=False)
        raise OIDCSessionEnded('user_mismatch')
    max_age = settings.OIDC_SESSION_MAX_AGE_SECONDS
    if max_age and row.created_at + timedelta(seconds=max_age) <= _now():
        end_session(row, 'max_age', delete_django_session=False)
        raise OIDCSessionEnded('max_age')
    return row


def renew_django_session(request: HttpRequest, row: OIDCSession) -> None:
    """Slide the Django session (cookie and stored expiry) at most once per interval."""
    session = request.session
    now_ts = int(time.time())
    try:
        last = int(session.get(constants.SESSION_RENEWED_AT, 0) or 0)
    except (TypeError, ValueError):
        last = 0
    if now_ts - last < settings.OIDC_SESSION_RENEW_INTERVAL_SECONDS:
        return
    session[constants.SESSION_RENEWED_AT] = now_ts  # modified: Django re-saves and re-sends the cookie
    OIDCSession.objects.filter(pk=row.pk).update(
        last_seen_at=_now(),
        django_session_key=session.session_key or row.django_session_key or '',
    )


# --- refresh --------------------------------------------------------------------------
#: After a refresh of a sign-in fails, this process does not try that sign-in again for
#: this many seconds. The requests that queued behind the failed attempt get its outcome at
#: once -- the still-usable token, else ``OIDCAuthUnavailable`` -- instead of each repeating
#: the call one after the other while Keycloak is slow or down.
REFRESH_RETRY_GAP_SECONDS = 2.0

_recent_failures: Dict[str, Tuple[float, str]] = {}  # str(pk) -> (monotonic time, category)


def _note_failure(pk: Any, category: str) -> None:
    now = time.monotonic()
    if len(_recent_failures) > 1024:
        for key, (at, _) in list(_recent_failures.items()):
            if now - at >= REFRESH_RETRY_GAP_SECONDS:
                _recent_failures.pop(key, None)
    _recent_failures[str(pk)] = (now, category)


def reset_refresh_failures() -> None:
    """Forget recent refresh failures (tests: ``FakeOP.install()`` and ``uninstall()`` call it)."""
    _recent_failures.clear()


def _recent_failure(pk: Any) -> Optional[str]:
    """The category of a refresh of this sign-in that failed less than the gap ago, if any."""
    entry = _recent_failures.get(str(pk))
    if entry is None:
        return None
    at, category = entry
    if 0 <= time.monotonic() - at < REFRESH_RETRY_GAP_SECONDS:
        return category
    return None


def ensure_fresh(row: OIDCSession, *, min_ttl: int = 0, wait: Optional[bool] = None) -> OIDCSession:
    """A row whose access token is valid for at least ``max(skew, min_ttl)`` seconds,
    refreshed if needed (single-flight). ``wait``: block for another refresher even when
    the current token is still usable (default: only when it is not)."""
    need = max(int(settings.OIDC_REFRESH_SKEW_SECONDS), int(min_ttl or 0))
    if row.ended_at is not None:
        raise OIDCSessionEnded(row.end_reason or 'ended')
    remaining = row.remaining(_now())
    if remaining > need:
        return row
    usable = remaining > settings.OIDC_ACCESS_MIN_USABLE_SECONDS
    if not row.refresh_token_enc:
        if usable:
            return row
        end_session(row, 'no_refresh_token')
        raise OIDCSessionEnded('no_refresh_token')
    if _in_caller_atomic(_db()):
        # refreshing here would hold the row lock until the caller commits and lose the
        # new tokens on rollback
        if usable:
            return row
        raise OIDCAuthUnavailable('refresh_in_atomic', retry_after=1)
    breaker = get_breaker()
    if breaker.is_open():
        if usable:
            return row
        raise OIDCAuthUnavailable('circuit_open', retry_after=breaker.retry_after())
    blocking = (not usable) if wait is None else bool(wait)
    lock = keyed_lock(('oidc-refresh', str(row.pk)))
    try:
        if blocking:
            acquired = lock.acquire(blocking=True, timeout=float(settings.OIDC_REFRESH_WAIT_SECONDS))
        else:
            acquired = lock.acquire(blocking=False)
        if not acquired:
            if usable:
                return row
            return _usable_or_unavailable(_reload(row.pk), 'lock_wait')
        current = _reload(row.pk)  # a leader of this process may have refreshed meanwhile
        if current.remaining(_now()) > need:
            return current
        failed = _recent_failure(current.pk)
        if failed is not None:
            # the attempt this request queued behind has just failed: its outcome stands
            return _usable_or_unavailable(current, failed)
        kind, value = _refresh_locked(current.pk, need, blocking)
        if kind == 'failed':
            _note_failure(current.pk, value)
        elif kind == 'fresh':
            _recent_failures.pop(str(current.pk), None)
    finally:
        lock.release()
    if kind == 'fresh':
        return value
    if kind == 'ended':
        raise OIDCSessionEnded(value)
    latest = _reload(row.pk)
    if latest.remaining(_now()) > need:
        return latest
    return _usable_or_unavailable(latest, value or kind)


def _refresh_locked(pk: Any, need: int, blocking: bool) -> Tuple[str, Any]:
    """Lock the row and refresh it. Returns (kind, value):
    ('fresh', row) | ('ended', reason) | ('failed', category) | ('unavailable', reason) | ('contended', None)."""
    alias = _db()
    connection = connections[alias]
    breaker = get_breaker()
    outcome: Tuple[str, Any] = ('contended', None)
    post_commit = []
    try:
        with transaction.atomic(using=alias):
            if connection.vendor == 'postgresql':
                with connection.cursor() as cursor:
                    timeout_ms = int(float(settings.OIDC_REFRESH_WAIT_SECONDS) * 1000)
                    cursor.execute("SELECT set_config('lock_timeout', %s, true)", [f'{timeout_ms}ms'])
            # NO KEY UPDATE where the database has it: the refresh changes no key, and a plain
            # FOR UPDATE would block every insert of a row referencing this sign-in (its foreign
            # key check takes FOR KEY SHARE) for the length of the provider call
            locked = OIDCSession.objects.using(alias).select_for_update(
                nowait=not blocking,
                no_key=getattr(connection.features, 'has_select_for_no_key_update', False),
            ).filter(pk=pk).first()
            now = _now()
            if locked is None:
                outcome = ('ended', 'missing_row')
            elif locked.ended_at is not None:
                outcome = ('ended', locked.end_reason or 'ended')
            elif locked.remaining(now) > need:
                outcome = ('fresh', locked)  # another process refreshed while we waited
            elif not locked.refresh_token_enc:
                outcome = ('unavailable', 'no_refresh_token')
            elif not breaker.try_acquire():
                outcome = ('unavailable', 'circuit_open')
            else:
                try:
                    refresh_token = locked.refresh_token or ''
                except TokenDecryptError:
                    breaker.release_probe()
                    locked.mark_ended('decrypt_failed', now)
                    locked.save(update_fields=END_FIELDS)
                    transaction.on_commit(partial(after_end, locked, 'decrypt_failed', True), using=alias)
                    outcome = ('ended', 'decrypt_failed')
                else:
                    result = token_endpoint.refresh(refresh_token)
                    if result.ok:
                        problem = _apply(locked, result.data, now)
                        if problem is None:
                            locked.save(update_fields=TOKEN_FIELDS)
                            breaker.success()
                            transaction.on_commit(partial(_after_refresh, locked, result.duration_ms), using=alias)
                            outcome = ('fresh', locked)
                        else:
                            breaker.failure(token_endpoint.CONFIG)
                            post_commit.append(partial(_refresh_failed, locked, token_endpoint.CONFIG, result, problem))
                            outcome = ('failed', token_endpoint.CONFIG)
                    elif result.category == token_endpoint.TERMINAL:
                        breaker.success()  # the OP answered properly
                        locked.mark_ended('refresh_invalid_grant', now)
                        locked.save(update_fields=END_FIELDS)
                        post_commit.append(partial(_refresh_failed, locked, token_endpoint.TERMINAL, result, None))
                        transaction.on_commit(partial(after_end, locked, 'refresh_invalid_grant', True), using=alias)
                        outcome = ('ended', 'refresh_invalid_grant')
                    else:
                        breaker.failure(result.category)
                        post_commit.append(partial(_refresh_failed, locked, result.category, result, None))
                        outcome = ('failed', result.category)
    except DatabaseError as e:
        if not _is_contention(e):
            raise
        outcome = ('contended', None)
    for callback in post_commit:
        callback()
    return outcome


def _apply(row: OIDCSession, data: Dict[str, Any], now: datetime) -> Optional[str]:
    """Write a refresh response onto the locked row. Returns a problem code instead of
    applying when the response does not belong to this sign-in."""
    access = data.get('access_token')
    if not isinstance(access, str) or not access:
        return 'no_access_token'
    access_sub = unverified_claims(access).get('sub')
    if access_sub and row.sub and access_sub != row.sub:
        return 'sub_mismatch'
    new_id = data.get('id_token')
    id_claims: Optional[Dict[str, Any]] = None
    if new_id:
        try:
            id_claims = validate_id_token(new_id)
        except OIDCTokenError:
            return 'invalid_id_token'
        except (OIDCAuthUnavailable, OIDCConfigurationError):
            id_claims = None  # keys unavailable right now: keep the previous id token
        if id_claims is not None and str(id_claims.get('sub')) != row.sub:
            return 'sub_mismatch'
    row.access_token = access
    row.access_expires_at = _access_expiry(data, access, now)
    new_refresh = data.get('refresh_token')
    if new_refresh:
        row.refresh_token = new_refresh
        row.refresh_expires_at = _refresh_expiry(data, new_refresh, now)
    elif 'refresh_expires_in' in data:
        row.refresh_expires_at = _refresh_expiry(data, None, now)
    if new_id and id_claims is not None:
        row.id_token = new_id
        row.sid = str(id_claims.get('sid') or row.sid or '')
        merged = dict(row.claims or {})
        merged.update(allowlisted(id_claims))
        row.claims = merged
    row.version = (row.version or 0) + 1
    row.last_refresh_at = now
    return None


def _after_refresh(row: OIDCSession, duration_ms: int) -> None:
    events.emit('oidc.session.refreshed', logging.DEBUG, session_id=str(row.pk), duration_ms=duration_ms)
    _send(signals.oidc_session_refreshed, session=row)


def _refresh_failed(row: OIDCSession, category: str, result: Any, problem: Optional[str]) -> None:
    level = {token_endpoint.TERMINAL: logging.INFO, token_endpoint.CONFIG: logging.ERROR}.get(category, logging.WARNING)
    events.emit(
        'oidc.session.refresh_failed', level,
        session_id=str(row.pk), category=category, status=result.status,
        oauth_error=result.oauth_error, description=problem or result.description, duration_ms=result.duration_ms,
    )
    if category != token_endpoint.TERMINAL:
        _send(signals.oidc_session_refresh_failed, session=row, category=category,
              status=result.status, oauth_error=result.oauth_error)


def _send(signal: Any, **kwargs: Any) -> None:
    for receiver, response in signal.send_robust(sender=OIDCSession, **kwargs):
        if isinstance(response, Exception):
            logger.warning("receiver %r of an OIDC signal failed: %s", receiver, type(response).__name__)


# --- public token access --------------------------------------------------------------
def access_token_for(session_id: Any, *, min_ttl: int = 0, wait: Optional[bool] = None) -> str:
    """A fresh access token of a sign-in, for in-process consumers (e.g. a dashboard proxy).

    Refreshes when fewer than ``max(OIDC_REFRESH_SKEW_SECONDS, min_ttl)`` seconds are left.
    Like a request, it waits for a refresh another thread is running only when the current
    token is no longer usable; a still-usable token is returned at once, even if it has less
    than ``min_ttl`` left. ``wait=True`` waits in any case, at most
    ``OIDC_REFRESH_WAIT_SECONDS``. Raises OIDCSessionEnded / OIDCAuthUnavailable."""
    try:
        row = _reload(session_id)
    except (ValidationError, ValueError):
        raise OIDCSessionEnded('missing_row')
    row = ensure_fresh(row, min_ttl=min_ttl, wait=wait)
    try:
        return row.access_token
    except TokenDecryptError:
        end_session(row, 'decrypt_failed')
        raise OIDCSessionEnded('decrypt_failed')


# --- ending ---------------------------------------------------------------------------
def end_session(row: OIDCSession, reason: str, *, delete_django_session: bool = True) -> bool:
    """End a sign-in once. Side effects (Django session deletion, socket notification,
    signal, optional revocation) run after commit. Returns False when it had ended already."""
    now = _now()
    updated = OIDCSession.objects.filter(pk=row.pk, ended_at__isnull=True).update(ended_at=now, end_reason=reason[:32])
    if row.ended_at is None:
        row.ended_at = now
        row.end_reason = reason[:32]
    if updated:
        transaction.on_commit(partial(after_end, row, reason, delete_django_session), using=_db())
    return bool(updated)


def end_sessions(*, user: Any = None, sub: Optional[str] = None, sid: Optional[str] = None,
                 issuer: Optional[str] = None, reason: str) -> int:
    """End every active sign-in matching the filters (at least one of user, sub, sid)."""
    if user is None and not sub and not sid:
        raise ValueError("end_sessions() needs user, sub or sid")
    qs = OIDCSession.objects.filter(ended_at__isnull=True)
    if user is not None:
        qs = qs.filter(user=user)
    if sub:
        qs = qs.filter(sub=sub)
    if sid:
        qs = qs.filter(sid=sid)
    if issuer:
        qs = qs.filter(issuer=issuer)
    return sum(1 for row in list(qs) if end_session(row, reason))


def after_end(row: OIDCSession, reason: str, delete_django_session: bool = True) -> None:
    """Side effects of an ended sign-in. Best effort: never raises."""
    if delete_django_session and row.django_session_key:
        try:
            engine = import_module(dj_settings.SESSION_ENGINE)
            engine.SessionStore(session_key=row.django_session_key).delete()
        except Exception as e:  # pragma: no cover - depends on the session backend
            logger.warning("could not delete the Django session of an ended sign-in: %s", type(e).__name__)
    try:
        from .contrib.channels import notify_session_ended
        notify_session_ended(row)
    except Exception as e:  # channels missing or no layer
        logger.debug("no socket notification for an ended sign-in: %s", type(e).__name__)
    if settings.OIDC_REVOKE_ON_LOGOUT and reason in ('logout', 'total_logout') and row.refresh_token_enc:
        try:
            token_endpoint.revoke(row.refresh_token or '')
        except Exception as e:
            logger.warning("refresh token revocation failed: %s", type(e).__name__)
    _send(signals.oidc_session_ended, session=row, reason=reason)
    events.emit('oidc.session.ended', logging.INFO, session_id=str(row.pk), user_id=row.user_id, reason=reason)


def logout_request(request: HttpRequest, reason: str) -> None:
    """Sign the current request out after its OIDC sign-in ended."""
    try:
        auth.logout(request)
    except Exception:  # pragma: no cover - defensive
        request.session.flush()
    events.emit('oidc.request.signed_out', logging.INFO, reason=reason)


# --- legacy (2.x) sessions ------------------------------------------------------------
def _drop_legacy_keys(session: Any) -> None:
    for key in constants.LEGACY_SESSION_KEYS:
        session.pop(key, None)


def migrate_legacy(request: HttpRequest) -> Optional[OIDCSession]:
    """Move the tokens of a 2.x session into a row (once, on its first 3.0 request)."""
    session = request.session
    user = getattr(request, 'user', None)
    access = session.get(constants.SESSION_ACCESS_TOKEN)
    if not isinstance(access, str) or not access or user is None or not user.is_authenticated:
        _drop_legacy_keys(session)
        return None
    lock = keyed_lock(('oidc-legacy', session.session_key or id(session)))
    try:
        lock.acquire(blocking=True, timeout=float(settings.OIDC_REFRESH_WAIT_SECONDS))
        refresh = session.get(constants.SESSION_REFRESH_TOKEN)
        refresh = refresh if isinstance(refresh, str) and refresh else None
        id_token = session.get(constants.SESSION_ID_TOKEN)
        id_token = id_token if isinstance(id_token, str) and id_token else None
        claims = unverified_claims(id_token) or unverified_claims(access)
        raw_expiry: Any = session.get(constants.SESSION_ACCESS_EXPIRES_AT)
        try:
            access_expires_at = _from_ts(float(raw_expiry))
        except (TypeError, ValueError, OverflowError, OSError):
            access_expires_at = _token_exp(access) or _now()
        issuer = str(claims.get('iss') or '')
        row = OIDCSession(
            user=user,
            django_session_key=session.session_key or '',
            issuer=issuer,
            client_id=settings.OIDC_RP_CLIENT_ID,
            sub=str(claims.get('sub') or ''),
            sid=str(claims.get('sid') or ''),
            claims=allowlisted(claims),
            access_expires_at=access_expires_at,
            refresh_expires_at=_token_exp(refresh) if refresh else None,
            migrated_from_legacy=True,
        )
        row.access_token = access
        row.refresh_token = refresh
        row.id_token = id_token
        row.save()
        _drop_legacy_keys(session)
        session[settings.OIDC_SESSION_KEY] = str(row.pk)
        events.emit('oidc.session.started', logging.INFO, session_id=str(row.pk), user_id=row.user_id, legacy=True)
        _send(signals.oidc_session_started, session=row, request=request)
        return row
    finally:
        lock.release()


__all__ = [
    'access_token_for',
    'after_end',
    'allowlisted',
    'attach_access_token',
    'create_session',
    'end_session',
    'end_sessions',
    'ensure_fresh',
    'has_legacy_keys',
    'load_for_request',
    'logout_request',
    'migrate_legacy',
    'renew_django_session',
]
