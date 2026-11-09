import os
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import (
    datetime,
    timezone,
)
from logging import getLogger
from re import search
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from urllib.parse import urlencode

import jwt
from django.contrib.auth import (
    BACKEND_SESSION_KEY,
    authenticate,
)
from django.core.cache import caches
from django.core.cache.backends.base import (
    BaseCache,
    InvalidCacheBackendError,
)
from django.core.exceptions import ImproperlyConfigured
from django.http import (
    HttpResponseRedirect,
    JsonResponse,
)
from django.http.request import HttpRequest
from django.http.response import HttpResponse
from django.urls import reverse
from django.utils.html import escape
from django.utils.module_loading import import_string
from requests import RequestException
from requests import post as request_post

from .auth import (
    AuthenticationBackend,
    BearerAuthenticationBackend,
)
from .conf import (
    constants,
    settings,
)
from .models import BlacklistedToken
from .utils import op_request_kwargs

logger = getLogger(__name__)

#: Session keys holding the OIDC tokens of the 2.x session layout.
TOKEN_SESSION_KEYS: Tuple[str, ...] = (
    constants.SESSION_ID_TOKEN,
    constants.SESSION_ACCESS_TOKEN,
    constants.SESSION_ACCESS_EXPIRES_AT,
    constants.SESSION_REFRESH_TOKEN,
    constants.SESSION_EXPIRES_AT,
)
#: OAuth2 errors of the token endpoint that mean the refresh token (or the OP session behind it) is gone for good.
TERMINAL_OAUTH_ERRORS = frozenset({'invalid_grant'})
#: Machine readable codes of the JSON 401 body.
NOT_AUTHENTICATED = 'not_authenticated'
SESSION_EXPIRED = 'session_expired'


class MiddlewareException(Exception):
    """
    The request needs a (new) authentication.

    - ``error``: the machine readable code answered to API clients (``not_authenticated`` or ``session_expired``).
    - ``reason``: why, for the logs.
    - ``end_session``: whether the Django session must be destroyed (only sessions holding OIDC tokens ever are).

    The message never contains token material.
    """
    def __init__(self, message: str, *, error: str = SESSION_EXPIRED, reason: Optional[str] = None, end_session: bool = True) -> None:
        super().__init__(message)
        self.error = error
        self.reason = reason or error
        self.end_session = end_session

    def __str__(self) -> str:
        return self.args[0]


class AuthUnavailable(Exception):
    """
    The OIDC provider could not give a definitive answer (unreachable, 5xx, 429, HTML page, configuration error, …)
    and the access token of the session has expired. The session is kept; the client should retry.
    """
    def __init__(self, category: str, retry_after: int) -> None:
        super().__init__(f"OIDC provider unavailable ({category})")
        self.category = category
        self.retry_after = retry_after


@dataclass
class RefreshOutcome:
    """Classified answer of the token endpoint to a refresh_token grant."""
    category: str  # 'ok', 'terminal', 'config' or 'transient'
    status: Optional[int] = None
    oauth_error: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


def classify_token_response(status: int, body: Any) -> Tuple[str, Optional[str]]:
    """
    Classify a token-endpoint answer on its OAuth2 ``error`` field, not on the status code.

    - ``ok``: a JSON object with an ``access_token``.
    - ``terminal``: ``invalid_grant`` (refresh token expired or revoked, OP session ended, user disabled):
      the session is over. An ``invalid_grant`` about the token issuer is a configuration problem, not the user's.
    - ``config``: any other OAuth2 error (``invalid_client``, ``unauthorized_client``, …): the session is kept.
    - ``transient``: anything else (429, 5xx, HTML pages from a proxy, a 2xx without token): the session is kept.
    """
    if not isinstance(body, dict):
        return 'transient', None
    oauth_error = body.get('error')
    if isinstance(oauth_error, str) and oauth_error:
        if oauth_error in TERMINAL_OAUTH_ERRORS:
            description = str(body.get('error_description') or '').lower()
            if 'issuer' in description:
                return 'config', oauth_error
            return 'terminal', oauth_error
        return 'config', oauth_error
    if 200 <= status < 300 and body.get('access_token'):
        return 'ok', None
    return 'transient', None


class _ProcessLocks:
    """Process-local locks keyed by name, forgotten once nobody holds or waits for them."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: Dict[str, List[Any]] = {}  # key -> [threading.Lock, users]

    def acquire(self, key: str) -> bool:
        with self._guard:
            entry = self._locks.setdefault(key, [threading.Lock(), 0])
            entry[1] += 1
        if entry[0].acquire(blocking=False):
            return True
        self._unref(key)
        return False

    def release(self, key: str) -> None:
        with self._guard:
            entry = self._locks.get(key)
        if entry is not None:
            entry[0].release()
            self._unref(key)

    def _unref(self, key: str) -> None:
        with self._guard:
            entry = self._locks.get(key)
            if entry is not None:
                entry[1] -= 1
                if entry[1] <= 0:
                    del self._locks[key]


_process_locks = _ProcessLocks()


class RefreshLock:
    """
    Single-flight lock for the refresh of one session.

    The lock lives in a Django cache shared by all processes and carries an owner token, so only its owner releases it.
    When the cache cannot be used (it raises, or ``django_redis`` with ``IGNORE_EXCEPTIONS`` answers ``None``), a
    process-local lock is used instead: a cache outage must never turn into a refresh stampede.
    """

    def __init__(self, cache: BaseCache, key: str, ttl: int) -> None:
        self.cache = cache
        self.key = key
        self.ttl = ttl
        self.token = secrets.token_hex(16)
        self.mode: Optional[str] = None  # 'shared' or 'local' once acquired

    def acquire(self) -> bool:
        try:
            added = self.cache.add(self.key, self.token, timeout=self.ttl)
        except Exception as e:  # any cache backend error
            logger.warning("oidc refresh lock: cache unavailable (%s), using a process-local lock", e.__class__.__name__)
            added = None
        if added is None:
            if _process_locks.acquire(self.key):
                self.mode = 'local'
                return True
            return False
        if added:
            self.mode = 'shared'
            return True
        return False

    def release(self) -> None:
        if self.mode == 'shared':
            try:
                if self.cache.get(self.key) == self.token:
                    self.cache.delete(self.key)
            except Exception as e:
                logger.debug("oidc refresh lock: release failed (%s)", e.__class__.__name__)
        elif self.mode == 'local':
            _process_locks.release(self.key)
        self.mode = None


GetResponseCallable = Callable[[HttpRequest], HttpResponse]
CheckFunctionCallable = Callable[[HttpRequest], None]

_RETRY_PAGE = (
    '<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="{retry_after}">'
    '<title>Sign-in temporarily unavailable</title></head><body><p>The sign-in service is temporarily unavailable. '
    'This page retries automatically in {retry_after} seconds.</p><p><a href="{url}">Retry now</a></p></body></html>'
)


_NAVIGATION_DESTINATIONS = ('document', 'iframe', 'frame', 'embed', 'object')


def _utc_now() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())


class Oauth2MiddlewareMixin:
    """
    Takes optionals token_type and check_function.
    Each request call trigger a call to process_request which uses check_function to verify if oauth2 tokens are still ok.

    - A ``MiddlewareException`` means the request needs a (new) authentication: navigations are redirected to the login,
      API and fetch/XHR requests get a JSON 401. Only sessions holding OIDC tokens are destroyed, never anonymous ones.
    - An ``AuthUnavailable`` means the OP could not answer: 503 with ``Retry-After``, the session is kept.
    """
    get_response: GetResponseCallable
    token_type: Optional[str]
    check_function: Optional[CheckFunctionCallable]
    exempt_urls: Tuple[str, ...]

    def __init__(self, get_response: GetResponseCallable, token_type: Optional[str], check_function: Optional[CheckFunctionCallable]) -> None:
        oidc_endpoints = tuple(
            f'^{reverse(url)}' for url in (
                constants.OIDC_URL_AUTHENTICATION_NAME,
                constants.OIDC_URL_CALLBACK_NAME,
                constants.OIDC_URL_LOGOUT_NAME,
                constants.OIDC_URL_TOTAL_LOGOUT_NAME,
                constants.OIDC_URL_LOGOUT_BY_OP_NAME,
            )
        )
        self.get_response = get_response
        self.token_type = token_type
        self.check_function = check_function
        self.exempt_urls = oidc_endpoints + tuple(str(p) for p in settings.OIDC_MIDDLEWARE_NO_AUTH_URL_PATTERNS)
        logger.debug(f"{self.exempt_urls=}")
        refresh_exempt_patterns = settings.OIDC_MIDDLEWARE_NO_REFRESH_URL_PATTERNS or ()
        self.refresh_exempt_urls = oidc_endpoints + tuple(str(p) for p in refresh_exempt_patterns)

    def __call__(self, request: HttpRequest) -> HttpResponse:
        response = self.process_request(request)
        return response or self.get_response(request)

    def is_oidc_enabled(self, request: HttpRequest) -> bool:
        auth_backend = None
        backend_session = request.session.get(BACKEND_SESSION_KEY)
        logger.debug(f"{backend_session=}")
        if backend_session and hasattr(request, 'user') and request.user.is_authenticated:
            auth_backend = import_string(backend_session)
        return issubclass(auth_backend, AuthenticationBackend) if auth_backend else False

    def is_refresh_exempt(self, request: HttpRequest) -> bool:
        return any(search(p, request.path) for p in self.refresh_exempt_urls)

    def is_refreshable_url(self, request: HttpRequest) -> bool:
        """
        Takes a request and returns whether it triggers a refresh examination
        :arg HttpRequest request:
        :returns: boolean
        """
        # Do not attempt to refresh the session if the OIDC backend is not used
        is_oidc_enabled = self.is_oidc_enabled(request)
        logger.debug(f"{is_oidc_enabled=}, {request.path=}")
        if not is_oidc_enabled:
            return False
        return not self.is_refresh_exempt(request)

    def has_oidc_session_data(self, request: HttpRequest) -> bool:
        """Whether the session holds OIDC tokens (2.x layout), i.e. whether ending the OIDC session means destroying it."""
        session = getattr(request, 'session', None)
        if session is None:
            return False
        return any(key in session for key in TOKEN_SESSION_KEYS)

    def is_oidc3_session(self, request: HttpRequest) -> bool:
        """A session written by the 3.0 layout (tokens in an OIDCSession row), which this version cannot use."""
        session = request.session
        return constants.SESSION_OIDC3_ID in session and not any(
            key in session for key in (constants.SESSION_ID_TOKEN, constants.SESSION_ACCESS_TOKEN, constants.SESSION_REFRESH_TOKEN)
        )

    def reject_oidc3_session(self, request: HttpRequest) -> None:
        """
        Rollback safety: a 3.0 session is treated as unauthenticated. Nothing is blacklisted and nothing is flushed,
        so the 3.0 session is still usable by a 3.0 process and a new login here simply adds the 2.x tokens.
        """
        if self.is_oidc3_session(request):
            raise MiddlewareException(
                "session written by a newer version of django-oauth2-codeflow",
                error=SESSION_EXPIRED, reason='unsupported_session_layout', end_session=False,
            )

    def check_blacklisted(self, request: HttpRequest) -> None:
        """
        Deprecated, no longer called by the middlewares: for cookie sessions the Django session is the credential,
        and checking the id token on every request turned concurrent token refreshes into logouts.
        """
        id_token = request.session.get(constants.SESSION_ID_TOKEN)
        if not id_token:
            return
        if BlacklistedToken.is_blacklisted(id_token):
            logger.debug("the session id token is blacklisted")
            raise MiddlewareException("token is blacklisted", reason='blacklisted')

    def get_param_url(self, request: HttpRequest, get_field: str, session_field: str) -> str:
        url = request.GET.get(get_field) if request.method == 'GET' else None
        if not url:
            url = request.session.get(session_field)
        if not url:
            url = '/'
        return request.build_absolute_uri(url)

    def get_next_url(self, request: HttpRequest) -> str:
        return self.get_param_url(request, settings.OIDC_REDIRECT_OK_FIELD_NAME, constants.SESSION_NEXT_URL)

    def get_failure_url(self, request: HttpRequest) -> str:
        return self.get_param_url(request, settings.OIDC_REDIRECT_ERROR_FIELD_NAME, constants.SESSION_FAIL_URL)

    def destroy_session(self, request: HttpRequest) -> None:
        # Use Django's session API to avoid response-phase SessionInterrupted races
        # when another request logs out/deletes the same session concurrently.
        request.session.flush()

    def is_explicit_navigation(self, request: HttpRequest) -> bool:
        """A page navigation announced by the browser (Fetch Metadata): it can follow a redirect to the login page."""
        if 'HTTP_AUTHORIZATION' in request.META:
            return False
        mode = request.headers.get('Sec-Fetch-Mode')
        if mode:
            return mode.lower() == 'navigate'
        dest = request.headers.get('Sec-Fetch-Dest')
        return dest is not None and dest.lower() in _NAVIGATION_DESTINATIONS

    def is_fetch_request(self, request: HttpRequest) -> bool:
        """
        Whether the request was made by script (fetch/XHR) or a non-browser API client rather than by a page navigation.
        Requests carrying no signal at all keep the historical navigation behaviour (redirect).
        """
        if 'HTTP_AUTHORIZATION' in request.META:
            return True
        mode = request.headers.get('Sec-Fetch-Mode')
        if mode:
            return mode.lower() != 'navigate'
        dest = request.headers.get('Sec-Fetch-Dest')
        if dest:
            return dest.lower() not in _NAVIGATION_DESTINATIONS
        if request.headers.get('X-Requested-With', '').lower() == 'xmlhttprequest':
            return True
        accept = request.headers.get('Accept', '')
        return 'application/json' in accept and 'text/html' not in accept

    def is_api_request(self, request: HttpRequest) -> bool:
        """Whether a JSON answer is expected rather than a redirect: explicit navigations are always redirected."""
        if self.is_explicit_navigation(request):
            return False
        if any(search(url_pattern, request.path) for url_pattern in settings.OIDC_MIDDLEWARE_API_URL_PATTERNS):
            return True
        return self.is_fetch_request(request)

    def login_url(self) -> str:
        return reverse(constants.OIDC_URL_AUTHENTICATION_NAME)

    def json_401(self, request: HttpRequest, error: str) -> JsonResponse:
        """JSON response with Unauthorized HTTP error: a machine readable code, never a message or a token"""
        response = JsonResponse({'error': error, 'login_url': self.login_url(), 'token_type': self.token_type}, status=401)
        scheme = 'Bearer' if 'HTTP_AUTHORIZATION' in request.META else 'Session'
        try:
            realm = settings.OIDC_RP_CLIENT_ID
        except ImproperlyConfigured:  # an answer must never fail on a missing setting
            realm = 'oidc'
        response['WWW-Authenticate'] = f'{scheme} realm="{realm}"'
        response['Cache-Control'] = 'no-store'
        return response

    def re_authent(self, request: HttpRequest, next_url: str, failure_url: str) -> HttpResponseRedirect:
        """Redirect to authentication page"""
        response = HttpResponseRedirect(self.login_url() + '?' + urlencode({
            settings.OIDC_REDIRECT_OK_FIELD_NAME: next_url,
            settings.OIDC_REDIRECT_ERROR_FIELD_NAME: failure_url,
        }))
        response['Cache-Control'] = 'no-store'
        return response

    def re_authent_or_401(self, request: HttpRequest, error: str, next_url: str, failure_url: str) -> Union[JsonResponse, HttpResponseRedirect]:
        if self.is_api_request(request):
            return self.json_401(request, error)
        else:
            return self.re_authent(request, next_url, failure_url)

    def auth_unavailable_response(self, request: HttpRequest, exc: AuthUnavailable) -> HttpResponse:
        """503: the OP could not answer and the access token has expired. The session is kept; the client retries."""
        retry_after = max(1, int(exc.retry_after))
        response: HttpResponse
        if self.is_api_request(request):
            response = JsonResponse({'error': 'auth_unavailable', 'retry_after': retry_after}, status=503)
        else:
            page = _RETRY_PAGE.format(retry_after=retry_after, url=escape(request.get_full_path()))
            response = HttpResponse(page, status=503, content_type='text/html; charset=utf-8')
        response['Retry-After'] = str(retry_after)
        response['Cache-Control'] = 'no-store'
        return response

    def process_request(self, request: HttpRequest) -> Optional[HttpResponse]:
        try:
            if self.check_function:
                self.check_function(request)
            return None
        except AuthUnavailable as e:
            # Same line format as codeflow 3.0's structured events, so one set of log-based metrics covers both
            logger.warning("event=oidc.session.refresh_failed category=%s path=%s", e.category, request.path)
            return self.auth_unavailable_response(request, e)
        except MiddlewareException as e:
            next_url, failure_url = self.get_next_url(request), self.get_failure_url(request)
            if e.end_session and self.has_oidc_session_data(request):
                logger.warning("event=oidc.session.ended reason=%s path=%s", e.reason, request.path)
                self.destroy_session(request)
            else:
                logger.debug("oidc: authentication required (reason=%s, path=%s)", e.reason, request.path)
            return self.re_authent_or_401(request, e.error, next_url, failure_url)


class LoginRequiredMiddleware(Oauth2MiddlewareMixin):
    """
    Force a user to be logged-in to access all pages not listed in OIDC_MIDDLEWARE_NO_AUTH_URL_PATTERNS.
    If OIDC_MIDDLEWARE_LOGIN_REQUIRED_REDIRECT is true (default), page navigations are redirected to the login page,
    fetch/XHR and API requests get a JSON 401.
    """
    def __init__(self, get_response: GetResponseCallable) -> None:
        super().__init__(get_response, 'id_token', self.check_login_required)

    def is_login_required_for_url(self, request: HttpRequest) -> bool:
        login_required_url = False
        for url_pattern in self.exempt_urls:
            if search(url_pattern, request.path):
                break
        else:
            login_required_url = True
        return login_required_url

    def is_api_request(self, request: HttpRequest) -> bool:
        if not settings.OIDC_MIDDLEWARE_LOGIN_REQUIRED_REDIRECT:
            return True
        if request.method in ('GET', 'HEAD') and not self.is_fetch_request(request):
            # a page navigation, even to an API URL: redirect to the login page
            return False
        return super().is_api_request(request)

    def check_login_required(self, request: HttpRequest) -> None:
        if hasattr(request, 'user') and request.user.is_authenticated:
            logger.debug("user is already authenticated")
            return
        if not self.is_login_required_for_url(request):
            logger.debug(f"{request.path} does not need authenticated user")
            return
        logger.debug(f"{request.path} needs an authenticated user")
        if not request.session.get(constants.SESSION_ID_TOKEN):
            try:
                user = authenticate(request)
            except Exception as e:
                raise MiddlewareException(str(e), error=NOT_AUTHENTICATED, reason='authenticate_error', end_session=False)
            if not user:
                raise MiddlewareException(
                    "id token is missing, user is not authenticated", error=NOT_AUTHENTICATED, reason='not_authenticated', end_session=False,
                )
        else:
            logger.debug("id token is present, authenticated user")


class RefreshAccessTokenMiddleware(Oauth2MiddlewareMixin):
    """
    Refreshes the access token with the OIDC RP shortly before it expires.

    - One refresh at a time per session (``RefreshLock``); concurrent requests use the current token while it is
      valid, otherwise wait for the leader's result. They never refresh in parallel with it.
    - Only ``invalid_grant`` ends the session. Any other failure keeps it: the still valid access token is served,
      otherwise a 503 asks the client to retry.
    - Every successful refresh moves the Django session idle timeout (``oidc_expires_at``) forward.
    """
    REFRESH_LOCK_TTL = 20
    REFRESH_WAIT_SECONDS = 5
    REFRESH_WAIT_STEP = 0.2
    #: Retry-After (seconds) when concurrent requests waited for a refresh that did not finish in time.
    REFRESH_IN_PROGRESS_RETRY_AFTER = 1

    def __init__(self, get_response: GetResponseCallable) -> None:
        super().__init__(get_response, 'access_token', self.check_access_token)

    @staticmethod
    def _expires_at(data: Any) -> float:
        try:
            return float(data.get(constants.SESSION_ACCESS_EXPIRES_AT) or 0)
        except (TypeError, ValueError):
            return 0.0

    def get_lock_cache(self) -> BaseCache:
        alias = settings.OIDC_REFRESH_LOCK_CACHE_ALIAS
        if alias:
            candidates = [alias, 'default']
        else:  # 2.0 behaviour, kept for compatibility: set OIDC_REFRESH_LOCK_CACHE_ALIAS instead
            candidates = ['redis' if os.getenv("DEPLOYMENT_ENVIRONMENT") else 'local', 'default']
        for candidate in candidates:
            try:
                return caches[candidate]
            except InvalidCacheBackendError:
                continue
        return caches['default']

    def _load_stored(self, request: HttpRequest) -> Optional[Dict[str, Any]]:
        """
        The session as currently stored (another request may have refreshed it), without touching ``request.session``.
        ``None`` when the stored session is gone (a concurrent request ended it).
        """
        session_key = request.session.session_key
        if not session_key:
            return dict(request.session.items())
        store = request.session.__class__(session_key=session_key)
        data = store.load()
        return dict(data) if data else None

    def _copy_tokens(self, request: HttpRequest, data: Dict[str, Any]) -> None:
        for key in TOKEN_SESSION_KEYS:
            if key in data:
                request.session[key] = data[key]

    def _is_newer(self, request: HttpRequest, data: Dict[str, Any], now: int) -> bool:
        return (
            data.get(constants.SESSION_ACCESS_TOKEN) != request.session.get(constants.SESSION_ACCESS_TOKEN)
            and self._expires_at(data) > now
        )

    def _wait_for_peer(self, request: HttpRequest) -> Optional[Dict[str, Any]]:
        """Poll the stored session (read only) until the refreshing request stored new tokens, or give up."""
        deadline = time.monotonic() + self.REFRESH_WAIT_SECONDS
        while time.monotonic() < deadline:
            time.sleep(self.REFRESH_WAIT_STEP)
            stored = self._load_stored(request)
            if stored is None:
                raise MiddlewareException("session ended by a concurrent request", reason='ended_by_concurrent_request')
            if self._is_newer(request, stored, _utc_now()):
                return stored
        return None

    def _serve_or_unavailable(self, request: HttpRequest, category: str, retry_after: Optional[int] = None) -> None:
        if self._expires_at(request.session) > _utc_now():
            logger.debug("oidc: serving the current access token (%s)", category)
            return
        raise AuthUnavailable(category, retry_after if retry_after is not None else settings.OIDC_RETRY_AFTER_SECONDS)

    def request_refresh(self, request: HttpRequest, refresh_token: str) -> RefreshOutcome:
        params = {
            "grant_type": "refresh_token",
            "client_id": settings.OIDC_RP_CLIENT_ID,
            "refresh_token": refresh_token,
        }
        if not settings.OIDC_RP_USE_PKCE or settings.OIDC_RP_FORCE_SECRET_WITH_PKCE:
            params["client_secret"] = settings.OIDC_RP_CLIENT_SECRET or ""
        try:
            resp = request_post(request.session[constants.SESSION_OP_TOKEN_URL], data=params, **op_request_kwargs())
        except RequestException as e:
            return RefreshOutcome('transient', oauth_error=e.__class__.__name__)
        try:
            body = resp.json()
        except ValueError:
            body = None
        category, oauth_error = classify_token_response(resp.status_code, body)
        return RefreshOutcome(category, status=resp.status_code, oauth_error=oauth_error, payload=body if category == 'ok' else None)

    def store_tokens(self, request: HttpRequest, result: Dict[str, Any], now: int) -> None:
        access_token = result["access_token"]
        expires_in = result.get("expires_in")
        if not isinstance(expires_in, (int, float)) or expires_in <= 0:
            try:
                expires_in = int(jwt.decode(access_token, options={'verify_signature': False}).get('exp', 0)) - now
            except jwt.PyJWTError:
                expires_in = 0
            if expires_in <= 0:
                expires_in = 60
        session = request.session
        id_token = result.get("id_token") or session.get(constants.SESSION_ID_TOKEN)
        if id_token:
            session[constants.SESSION_ID_TOKEN] = id_token
        else:
            session.pop(constants.SESSION_ID_TOKEN, None)
        session[constants.SESSION_ACCESS_TOKEN] = access_token
        session[constants.SESSION_ACCESS_EXPIRES_AT] = now + int(expires_in)
        session[constants.SESSION_REFRESH_TOKEN] = result.get("refresh_token") or session[constants.SESSION_REFRESH_TOKEN]
        # sliding idle timeout of the Django side of the session
        session[constants.SESSION_EXPIRES_AT] = now + settings.OIDC_MIDDLEWARE_SESSION_TIMEOUT_SECONDS
        session.save()

    def check_access_token(self, request: HttpRequest) -> None:
        if not self.is_refreshable_url(request):
            return
        self.reject_oidc3_session(request)
        if constants.SESSION_REFRESH_TOKEN not in request.session:
            return
        now = _utc_now()
        expires_at = self._expires_at(request.session)
        if expires_at - now > settings.OIDC_REFRESH_SKEW_SECONDS:
            logger.debug('access token is still valid')
            return
        token_still_valid = expires_at > now
        lock = RefreshLock(self.get_lock_cache(), f"oidc:refresh:{request.session.session_key}", self.REFRESH_LOCK_TTL)
        if not lock.acquire():
            # another request refreshes this session: never refresh in parallel with it
            if token_still_valid:
                return
            fresh = self._wait_for_peer(request)
            if fresh is not None:
                self._copy_tokens(request, fresh)
                return
            self._serve_or_unavailable(request, 'refresh_in_progress', self.REFRESH_IN_PROGRESS_RETRY_AFTER)
            return
        try:
            stored = self._load_stored(request)
            if stored is None:
                raise MiddlewareException("session ended by a concurrent request", reason='ended_by_concurrent_request')
            if self._is_newer(request, stored, now) and self._expires_at(stored) - now > settings.OIDC_REFRESH_SKEW_SECONDS:
                self._copy_tokens(request, stored)
                return
            refresh_token = stored.get(constants.SESSION_REFRESH_TOKEN) or request.session[constants.SESSION_REFRESH_TOKEN]
            outcome = self.request_refresh(request, refresh_token)
            if outcome.category == 'ok' and outcome.payload is not None:
                self.store_tokens(request, outcome.payload, _utc_now())
                logger.debug('access token refreshed')
                return
            if outcome.category == 'terminal':
                raise MiddlewareException("the OIDC provider rejected the refresh token", reason=f"refresh_{outcome.oauth_error}")
            log = logger.error if outcome.category == 'config' else logger.warning
            log(
                "oidc: token refresh failed (category=%s, status=%s, error=%s); the session is kept",
                outcome.category, outcome.status, outcome.oauth_error,
            )
            self._serve_or_unavailable(request, outcome.category)
        finally:
            lock.release()


class RefreshSessionMiddleware(Oauth2MiddlewareMixin):
    """
    Checks if the session expired.
    """
    MIN_SECONDS = 10

    def __init__(self, get_response: GetResponseCallable) -> None:
        if not (self.MIN_SECONDS < settings.OIDC_MIDDLEWARE_SESSION_TIMEOUT_SECONDS < settings.SESSION_COOKIE_AGE):
            raise ImproperlyConfigured(
                "OIDC_MIDDLEWARE_SESSION_TIMEOUT_SECONDS should be less than SESSION_COOKIE_AGE"
                f" and more than {self.MIN_SECONDS} seconds"
            )
        super().__init__(get_response, 'refresh_token', self.check_session)

    def check_session(self, request: HttpRequest) -> None:
        if not self.is_refreshable_url(request):
            logger.debug(f"{request.path} is not refreshable")
            return
        self.reject_oidc3_session(request)
        logger.debug(f"{request.path} is refreshable")
        utc_expiration = request.session.get(constants.SESSION_EXPIRES_AT)
        if not utc_expiration:
            msg = f"No {constants.SESSION_EXPIRES_AT} parameter in the backend session"
            logger.debug(msg)
            raise MiddlewareException(msg, reason='missing_session_expiry')
        utc_now = datetime.now(tz=timezone.utc).timestamp()
        if utc_expiration > utc_now:
            # The session is still valid, so we don't have to do anything.
            logger.debug(
                'session is still valid (%s > %s)',
                datetime.fromtimestamp(utc_expiration).strftime('%d/%m/%Y, %H:%M:%S'),
                datetime.fromtimestamp(utc_now).strftime('%d/%m/%Y, %H:%M:%S'),
            )
            return
        # The session has expired, an authentication is now required.
        # The id token can no longer be used as a bearer token.
        id_token = request.session.get(constants.SESSION_ID_TOKEN)
        if id_token:
            BlacklistedToken.blacklist(id_token)
        msg = "Session has expired"
        logger.debug(msg)
        raise MiddlewareException(msg, reason='session_idle_timeout')


class BearerAuthMiddleware(Oauth2MiddlewareMixin):
    """
    Inject User in request if authenticate from header.
    Only `Authorization` headers using the `OIDC_AUTHORIZATION_HEADER_PREFIX` scheme are considered.
    """
    def __init__(self, get_response: GetResponseCallable) -> None:
        super().__init__(get_response, None, None)

    def process_request(self, request: HttpRequest) -> Optional[HttpResponse]:
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith(f"{settings.OIDC_AUTHORIZATION_HEADER_PREFIX} "):
            user = BearerAuthenticationBackend().authenticate(request)
            if user:
                request.user = user  # type: ignore[assignment]
                if not request.session.session_key:
                    # ensure request.session.session_key exists
                    request.session.save()
        return None
