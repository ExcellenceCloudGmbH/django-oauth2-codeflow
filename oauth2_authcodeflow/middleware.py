from datetime import (
    datetime,
    timezone,
)
from logging import getLogger
from re import search
from typing import (
    Callable,
    Optional,
    Tuple,
    Union,
)
from urllib.parse import urlencode

from django.contrib.auth import (
    BACKEND_SESSION_KEY,
    authenticate,
)
from django.contrib.sessions.models import Session
from django.core.exceptions import ImproperlyConfigured
from django.http import (
    HttpResponseRedirect,
    JsonResponse,
)
from django.http.request import HttpRequest
from django.http.response import HttpResponse
from django.urls import reverse
from django.utils.module_loading import import_string
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
import time
from django.core.cache import caches
import os
import random

logger = getLogger(__name__)


class MiddlewareException(Exception):
    def __str__(self):
        return self.args[0]


GetResponseCallable = Callable[[HttpRequest], HttpResponse]
CheckFunctionCallable = Callable[[HttpRequest], None]


class Oauth2MiddlewareMixin:
    """
    Takes optionals token_type  and check_function.
    Each request call trigger a call to process_request which uses check_function to verify if oauth2 tokens are still ok.
    If not, a MiddlewareException should be raised and a redirection to login is realized or a json error returned.
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

        self.refresh_exempt_urls = oidc_endpoints + tuple(
            str(p) for p in getattr(settings, "OIDC_MIDDLEWARE_NO_REFRESH_URL_PATTERNS", ())
        )

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

    def is_refresh_exempt(self, request) -> bool:
        return any(search(p, request.path) for p in self.refresh_exempt_urls)

    def is_refreshable_url(self, request: HttpRequest) -> bool:
        """
        Takes a request and returns whether it triggers a refresh examination
        :arg HttpRequest request:
        :returns: boolean
        """
        # Do not attempt to refresh the session if the OIDC backend is not used
        is_oidc_enabled = self.is_oidc_enabled(request)
        logger.debug(f"{is_oidc_enabled=}, {request.path=}, {self.exempt_urls=}")
        if not is_oidc_enabled:
            return False
        # IMPORTANT: only skip refresh for refresh_exempt_urls (not login_exempt_urls)
        return not self.is_refresh_exempt(request)

    def _reload_session_from_store(self, request):
        SessionStore = request.session.__class__
        fresh = SessionStore(session_key=request.session.session_key)
        fresh.load()
        # copy relevant keys
        for k in (
            constants.SESSION_ID_TOKEN,
            constants.SESSION_ACCESS_TOKEN,
            constants.SESSION_ACCESS_EXPIRES_AT,
            constants.SESSION_REFRESH_TOKEN,
            constants.SESSION_EXPIRES_AT,
        ):
            if k in fresh:
                request.session[k] = fresh[k]

    def check_blacklisted(self, request):
        if constants.SESSION_ID_TOKEN not in request.session:
            return

        tok = request.session.get(constants.SESSION_ID_TOKEN)
        if not tok:
            return

        if not BlacklistedToken.is_blacklisted(tok):
            return

        # Token appears blacklisted — could be stale session snapshot in this request.
        # Reload once from the session store and re-check.
        try:
            self._reload_session_from_store(request)
            new_tok = request.session.get(constants.SESSION_ID_TOKEN)
            if new_tok and new_tok != tok and not BlacklistedToken.is_blacklisted(new_tok):
                return  # session was updated by another request; proceed safely
        except Exception:
            # if reload fails, fall back to original behavior
            pass

        raise MiddlewareException(f"token is blacklisted")

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
        try:
            Session.objects.get(session_key=request.session.session_key).delete()
        except Session.DoesNotExist:
            pass

    def is_api_request(self, request: HttpRequest) -> bool:
        return any(
            search(url_pattern, request.path)
            for url_pattern in settings.OIDC_MIDDLEWARE_API_URL_PATTERNS
        )

    def json_401(self, request: HttpRequest, error: str) -> JsonResponse:
        """Return JSON response with Unauthorized HTTP error"""
        return JsonResponse({'error': error, 'token_type': self.token_type}, status=401)

    def re_authent(self, request: HttpRequest, next_url: str, failure_url: str) -> HttpResponseRedirect:
        """Redirect to authentication page"""
        return HttpResponseRedirect(reverse(constants.OIDC_URL_AUTHENTICATION_NAME) + '?' + urlencode({
            settings.OIDC_REDIRECT_OK_FIELD_NAME: next_url,
            settings.OIDC_REDIRECT_ERROR_FIELD_NAME: failure_url,
        }))

    def re_authent_or_401(self, request: HttpRequest, error: str, next_url: str, failure_url: str) -> Union[JsonResponse, HttpResponseRedirect]:
        if self.is_api_request(request):
            return self.json_401(request, error)
        else:
            return self.re_authent(request, next_url, failure_url)

    def process_request(self, request: HttpRequest) -> Optional[HttpResponse]:
        try:
            self.check_blacklisted(request)
            if self.check_function:
                self.check_function(request)
            return None
        except MiddlewareException as e:
            next_url, failure_url = self.get_next_url(request), self.get_failure_url(request)
            self.destroy_session(request)
            return self.re_authent_or_401(request, str(e), next_url, failure_url)


class LoginRequiredMiddleware(Oauth2MiddlewareMixin):
    """
    Force a user to be logged-in to access all pages not listed in OIDC_MIDDLEWARE_NO_AUTH_URL_PATTERNS.
    If OIDC_MIDDLEWARE_LOGIN_REQUIRED_REDIRECT is true (default), then redirect to login page if not authenticated.
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
        if settings.OIDC_MIDDLEWARE_LOGIN_REQUIRED_REDIRECT:
            if request.method == 'GET':
                # force redirect on GET request even if it’s a API request
                return False
            else:
                return super().is_api_request(request)
        else:
            return True

    def check_login_required(self, request: HttpRequest) -> None:
        if hasattr(request, 'user') and request.user.is_authenticated:
            logger.debug("user is already authenticated")
            return
        if not self.is_login_required_for_url(request):
            logger.debug(f"{request.path} does not need authenticated user")
            return
        logger.debug(f"{request.path} needs an authenticated user")
        if constants.SESSION_ID_TOKEN not in request.session:
            try:
                user = authenticate(request)
            except Exception as e:
                raise MiddlewareException(str(e))
            if not user:
                raise MiddlewareException("id token is missing, user is not authenticated")
        else:
            logger.debug("id token is present, authenticated user")


class RefreshAccessTokenMiddleware(Oauth2MiddlewareMixin):
    # Refresh slightly before expiry to avoid stampede
    TOKEN_SKEW_SECONDS = 30

    # Network timeout to token endpoint
    REFRESH_HTTP_TIMEOUT = 10

    # Lock settings
    REFRESH_LOCK_TTL = 60  # must be > REFRESH_HTTP_TIMEOUT, ideally 3-6x
    REFRESH_WAIT_SECONDS = 65  # TTL + buffer

    # Backoff settings for followers
    BASE_SLEEP = 0.05
    MAX_SLEEP = 0.8

    def _reload_session_from_store(self, request):
        SessionStore = request.session.__class__
        fresh = SessionStore(session_key=request.session.session_key)
        fresh.load()
        for k in (
                constants.SESSION_ID_TOKEN,
                constants.SESSION_ACCESS_TOKEN,
                constants.SESSION_ACCESS_EXPIRES_AT,
                constants.SESSION_REFRESH_TOKEN,
                constants.SESSION_EXPIRES_AT,
        ):
            if k in fresh:
                request.session[k] = fresh[k]

    def _is_access_valid(self, request) -> bool:
        exp = request.session.get(constants.SESSION_ACCESS_EXPIRES_AT)
        if not exp:
            return False
        now = int(datetime.now(tz=timezone.utc).timestamp())
        return exp > (now + self.TOKEN_SKEW_SECONDS)

    def _do_refresh_as_leader(self, request):
        now = int(datetime.now(tz=timezone.utc).timestamp())

        params = {
            "grant_type": "refresh_token",
            "client_id": settings.OIDC_RP_CLIENT_ID,
            "refresh_token": request.session[constants.SESSION_REFRESH_TOKEN],
        }
        if not settings.OIDC_RP_USE_PKCE or settings.OIDC_RP_FORCE_SECRET_WITH_PKCE:
            params["client_secret"] = settings.OIDC_RP_CLIENT_SECRET or ""

        resp = request_post(
            request.session[constants.SESSION_OP_TOKEN_URL],
            data=params,
            timeout=self.REFRESH_HTTP_TIMEOUT,
        )

        if resp.status_code != 200:
            # Important: another request may already have refreshed in parallel (rotation race)
            self._reload_session_from_store(request)
            if self._is_access_valid(request):
                return
            raise MiddlewareException(resp.text)

        result = resp.json()
        access_token = result["access_token"]
        expires_in = int(result["expires_in"])

        new_id_token = result.get("id_token", request.session.get(constants.SESSION_ID_TOKEN))
        new_refresh_token = result.get("refresh_token", request.session.get(constants.SESSION_REFRESH_TOKEN))

        # 1) Capture the old id_token BEFORE overwriting
        old_id_token = request.session.get(constants.SESSION_ID_TOKEN)

        # 2) Write the new tokens into session
        request.session[constants.SESSION_ID_TOKEN] = new_id_token
        request.session[constants.SESSION_ACCESS_TOKEN] = access_token
        request.session[constants.SESSION_ACCESS_EXPIRES_AT] = utc_now + expires_in
        request.session[constants.SESSION_REFRESH_TOKEN] = new_refresh_token

        # 3) Persist session FIRST (so other concurrent requests can reload and see the new token)
        request.session.save()

        # 4) Only AFTER saving, blacklist the old token (if it actually changed)
        if old_id_token and new_id_token and new_id_token != old_id_token:
            BlacklistedToken.blacklist(old_id_token)

    def check_access_token(self, request):
        if not self.is_refreshable_url(request):
            return
        if constants.SESSION_REFRESH_TOKEN not in request.session:
            return

        # fast path
        if self._is_access_valid(request):
            return

        # ensure session key exists
        if not request.session.session_key:
            request.session.save()
        session_key = request.session.session_key

        # use dedicated lock cache if you add it, else "redis"
        lock_cache = "redis" if os.getenv("DEPLOYMENT_ENVIRONMENT") else "local"

        lock_key = f"oidc:refresh_lock:{session_key}"

        # "token" is not used for CAS delete here (we rely on TTL),
        # but can be useful for debugging.
        token = f"{random.randint(1, 10 ** 9)}"

        # Try to become leader
        if lock_cache.add(lock_key, token, timeout=self.REFRESH_LOCK_TTL):
            # leader
            self._reload_session_from_store(request)
            if self._is_access_valid(request):
                return

            self._do_refresh_as_leader(request)

            # DO NOT delete lock (safe). Let TTL expire naturally.
            return

        # follower: wait for leader to finish (or lock TTL to expire)
        deadline = time.monotonic() + self.REFRESH_WAIT_SECONDS
        sleep = self.BASE_SLEEP

        while time.monotonic() < deadline:
            time.sleep(sleep + random.uniform(0, sleep * 0.3))  # jitter
            self._reload_session_from_store(request)

            if self._is_access_valid(request):
                return

            # If lock expired (leader died), try to take over
            if lock_cache.get(lock_key) is None:
                if lock_cache.add(lock_key, token, timeout=self.REFRESH_LOCK_TTL):
                    self._reload_session_from_store(request)
                    if self._is_access_valid(request):
                        return
                    self._do_refresh_as_leader(request)
                    return

            sleep = min(self.MAX_SLEEP, sleep * 1.6)

        raise MiddlewareException("Token refresh did not complete in time")
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
        logger.debug(f"{request.path} is refreshable")
        utc_expiration = request.session.get(constants.SESSION_EXPIRES_AT)
        if not utc_expiration:
            msg = f"No {constants.SESSION_EXPIRES_AT} parameter in the backend session"
            logger.debug(msg)
            raise MiddlewareException(msg)
        utc_now = datetime.now(tz=timezone.utc).timestamp()
        if utc_expiration > utc_now:
            # The session is still valid, so we don't have to do anything.
            logger.debug(
                'session is still valid (%s > %s)',
                datetime.fromtimestamp(utc_expiration).strftime('%d/%m/%Y, %H:%M:%S'),
                datetime.fromtimestamp(utc_now).strftime('%d/%m/%Y, %H:%M:%S'),
            )
            return
        # The session has expired, an authentication is now required
        # Blacklist the current id token
        BlacklistedToken.blacklist(request.session[constants.SESSION_ID_TOKEN])
        msg = "Session has expired"
        logger.debug(msg)
        raise MiddlewareException(msg)


class BearerAuthMiddleware(Oauth2MiddlewareMixin):
    """
    Inject User in request if authenticate from header.
    """
    def __init__(self, get_response: GetResponseCallable) -> None:
        super().__init__(get_response, None, None)

    def process_request(self, request: HttpRequest) -> Optional[HttpResponse]:
        if 'Authorization' in request.headers:
            user = BearerAuthenticationBackend().authenticate(request)
            if user:
                request.user = user
                if not request.session.session_key:
                    # ensure request.session.session_key exists
                    request.session.save()
        return None
