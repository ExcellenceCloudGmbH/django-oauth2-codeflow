"""Sign-in, callback, logout and back-channel logout views.

- Pending logins are kept per ``state`` (several tabs or frames can sign in at once).
- PKCE (S256), ``state`` and ``nonce`` are always sent; ``prompt`` is passed through
  (``prompt=none`` enables silent re-authentication).
- ``next`` / ``fail`` must be relative or on an allowed host (no open redirect).
- A callback with ``?error=`` never signs anyone out.
- Back-channel logout follows OpenID Connect Back-Channel Logout 1.0.
"""
import hashlib
import logging
import secrets
import time
from base64 import urlsafe_b64encode
from typing import (
    Any,
    Dict,
    Optional,
)
from urllib.parse import urlencode

from django.contrib import auth
from django.core.cache import caches
from django.core.exceptions import (
    DisallowedHost,
    ValidationError,
)
from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseNotFound,
    HttpResponseRedirect,
    JsonResponse,
)
from django.urls import reverse
from django.utils.crypto import get_random_string
from django.utils.decorators import method_decorator
from django.utils.html import escape
from django.utils.http import url_has_allowed_host_and_scheme
from django.views import View
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_exempt

from . import (
    constants,
    events,
    responses,
    sessions,
    signals,
    token_endpoint,
)
from .conf import settings
from .crypto import TokenDecryptError
from .exceptions import (
    OIDCAuthUnavailable,
    OIDCConfigurationError,
    OIDCTokenError,
)
from .jwt_validation import (
    validate_id_token,
    validate_logout_token,
)
from .models import OIDCSession
from .provider import get_provider
from .users import provision_user

logger = logging.getLogger(__name__)

PROMPT_VALUES = frozenset({'none', 'login', 'consent', 'select_account'})


def _allowed_hosts(request: HttpRequest) -> set:
    hosts = set(settings.OIDC_ALLOWED_REDIRECT_HOSTS or [])
    try:
        hosts.add(request.get_host())
    except DisallowedHost:
        pass
    return hosts


def safe_redirect_target(request: HttpRequest, target: Optional[str]) -> Optional[str]:
    """``target`` when it is relative or on an allowed host, else None."""
    if not target:
        return None
    if url_has_allowed_host_and_scheme(target, allowed_hosts=_allowed_hosts(request), require_https=request.is_secure()):
        return target
    return None


def _with_query(url: str, params: Dict[str, Any]) -> str:
    return f"{url}{'&' if '?' in url else '?'}{urlencode(params)}"


def _no_cache(response: HttpResponse) -> HttpResponse:
    response['Cache-Control'] = 'no-store'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response


def _bad_redirect() -> HttpResponse:
    return _no_cache(HttpResponseBadRequest(b'Redirect target not allowed.'))


def _current_row(request: HttpRequest) -> Optional[OIDCSession]:
    key = request.session.get(settings.OIDC_SESSION_KEY)
    if not key:
        return None
    try:
        return OIDCSession.objects.filter(pk=key, ended_at__isnull=True).first()
    except (ValidationError, ValueError):
        return None


def _pending_logins(request: HttpRequest) -> Dict[str, Dict[str, Any]]:
    data = request.session.get(constants.SESSION_PENDING_LOGINS)
    if not isinstance(data, dict):
        return {}
    oldest = time.time() - settings.OIDC_LOGIN_STATE_TTL_SECONDS
    return {state: entry for state, entry in data.items() if isinstance(entry, dict) and entry.get('ts', 0) >= oldest}


@method_decorator(never_cache, name='dispatch')
class _OIDCView(View):
    def dispatch(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        return _no_cache(super().dispatch(request, *args, **kwargs))


class AuthenticateView(_OIDCView):
    """Start a sign-in: redirect to the OP authorization endpoint."""
    http_method_names = ['get']

    def get(self, request: HttpRequest) -> HttpResponse:
        next_url = safe_redirect_target(request, request.GET.get(settings.OIDC_REDIRECT_OK_FIELD_NAME) or '/')
        if next_url is None:
            return _bad_redirect()
        fail_url = None
        fail_raw = request.GET.get(settings.OIDC_REDIRECT_ERROR_FIELD_NAME)
        if fail_raw:
            fail_url = safe_redirect_target(request, fail_raw)
            if fail_url is None:
                return _bad_redirect()
        try:
            md = get_provider().metadata()
        except OIDCAuthUnavailable as e:
            return responses.auth_unavailable(request, e)
        except OIDCConfigurationError:
            return responses.auth_unavailable(request, OIDCAuthUnavailable('configuration'))

        size = max(32, settings.OIDC_RANDOM_SIZE)
        state = get_random_string(size)
        nonce = get_random_string(size)
        verifier = secrets.token_urlsafe(64) if settings.OIDC_RP_USE_PKCE else None
        pending = _pending_logins(request)
        pending[state] = {'nonce': nonce, 'verifier': verifier, 'next': next_url, 'fail': fail_url, 'ts': int(time.time())}
        while len(pending) > max(1, settings.OIDC_LOGIN_MAX_PENDING):
            oldest = min(pending, key=lambda s: pending[s].get('ts', 0))
            pending.pop(oldest)
        request.session[constants.SESSION_PENDING_LOGINS] = pending

        params: Dict[str, Any] = {
            'response_type': 'code',
            'client_id': settings.OIDC_RP_CLIENT_ID,
            'scope': ' '.join(settings.OIDC_RP_SCOPES),
            'redirect_uri': request.build_absolute_uri(reverse(constants.OIDC_URL_CALLBACK_NAME)),
            'state': state,
            'nonce': nonce,
        }
        if verifier:
            digest = hashlib.sha256(verifier.encode('ascii')).digest()
            params['code_challenge'] = urlsafe_b64encode(digest).decode('ascii').rstrip('=')
            params['code_challenge_method'] = 'S256'
        prompt = request.GET.get('prompt', '')
        if prompt and all(p in PROMPT_VALUES for p in prompt.split()):
            params['prompt'] = prompt
        elif settings.OIDC_RP_FORCE_CONSENT_PROMPT:
            params['prompt'] = 'consent'
        login_hint = request.GET.get('login_hint')
        if login_hint and len(login_hint) <= 256:
            params['login_hint'] = login_hint
        if md.claims_parameter_supported:
            import json
            claims: Dict[str, Any] = {}
            if settings.OIDC_RP_USERINFO_CLAIMS:
                claims['userinfo'] = settings.OIDC_RP_USERINFO_CLAIMS
            if settings.OIDC_RP_TOKEN_CLAIMS:
                claims['id_token'] = settings.OIDC_RP_TOKEN_CLAIMS
            if claims:
                params['claims'] = json.dumps(claims)
        return HttpResponseRedirect(_with_query(md.authorization_endpoint, params))


class CallbackView(_OIDCView):
    """Redirect target of the OP, for sign-ins and for RP-initiated logouts."""
    http_method_names = ['get']

    def get(self, request: HttpRequest) -> HttpResponse:
        state = request.GET.get('state') or ''
        pending_logout = request.session.get(constants.SESSION_PENDING_LOGOUT)
        if pending_logout and state and 'code' not in request.GET and 'error' not in request.GET:
            if isinstance(pending_logout, dict) and secrets.compare_digest(str(pending_logout.get('state', '')), state):
                request.session.pop(constants.SESSION_PENDING_LOGOUT, None)
                return HttpResponseRedirect(pending_logout.get('next') or '/')

        pending = _pending_logins(request)
        entry = pending.pop(state, None) if state else None
        request.session[constants.SESSION_PENDING_LOGINS] = pending
        if entry is None:
            return self.failure(request, None, 'invalid_state')
        fail_url = entry.get('fail')
        if request.GET.get('error'):
            # e.g. login_required after prompt=none: whoever is signed in stays signed in
            return self.failure(request, fail_url, str(request.GET['error'])[:64])
        code = request.GET.get('code')
        if not code:
            return self.failure(request, fail_url, 'missing_code')

        redirect_uri = request.build_absolute_uri(reverse(constants.OIDC_URL_CALLBACK_NAME))
        try:
            outcome = token_endpoint.exchange_code(code, redirect_uri, entry.get('verifier'))
        except (OIDCAuthUnavailable, OIDCConfigurationError):
            return self.failure(request, fail_url, 'auth_unavailable')
        if not outcome.ok:
            events.emit('oidc.login.failed', logging.WARNING, step='token_exchange', category=outcome.category,
                        status=outcome.status, oauth_error=outcome.oauth_error)
            error = 'auth_unavailable' if outcome.category == token_endpoint.TRANSIENT else 'token_exchange_failed'
            return self.failure(request, fail_url, error)
        data = outcome.data
        id_token = data.get('id_token')
        if not id_token:
            return self.failure(request, fail_url, 'missing_id_token')
        try:
            id_claims = validate_id_token(id_token, nonce=entry.get('nonce'))
        except OIDCTokenError as e:
            events.emit('oidc.login.failed', logging.WARNING, step='id_token', reason=e.reason)
            return self.failure(request, fail_url, 'invalid_id_token')
        except (OIDCAuthUnavailable, OIDCConfigurationError):
            return self.failure(request, fail_url, 'auth_unavailable')

        userinfo = token_endpoint.fetch_userinfo(data['access_token']) if settings.OIDC_OP_FETCH_USER_INFO else None
        if userinfo and str(userinfo.get('sub')) != str(id_claims.get('sub')):
            userinfo = None  # OIDC Core 5.3.2: userinfo of another subject must not be used
        claims = dict(id_claims)
        if userinfo:
            claims.update({k: v for k, v in userinfo.items() if k != 'sub'})
        try:
            user = provision_user(claims, request=request, access_token=data['access_token'])
        except OIDCTokenError:
            return self.failure(request, fail_url, 'missing_claims')
        if user is None:
            return self.failure(request, fail_url, 'user_not_allowed')
        if not getattr(user, 'is_active', True):
            return self.failure(request, fail_url, 'user_inactive')

        previous = _current_row(request)
        if previous is not None:
            sessions.end_session(previous, 'superseded', delete_django_session=False)
        row = sessions.create_session(user=user, token_data=data, id_claims=id_claims, extra_claims=userinfo)
        request._oidc_row = row  # type: ignore[attr-defined]  # user_logged_in receivers may ask for the token
        auth.login(request, user, backend=constants.AUTH_BACKEND_PATH)
        request.session[settings.OIDC_SESSION_KEY] = str(row.pk)
        request.session[constants.SESSION_RENEWED_AT] = int(time.time())
        if not request.session.session_key:
            request.session.save()
        row.django_session_key = request.session.session_key or ''
        OIDCSession.objects.filter(pk=row.pk).update(django_session_key=row.django_session_key)
        events.emit('oidc.session.started', logging.INFO, session_id=str(row.pk), user_id=row.user_id)
        sessions._send(signals.oidc_session_started, session=row, request=request)
        return HttpResponseRedirect(entry.get('next') or '/')

    def failure(self, request: HttpRequest, fail_url: Optional[str], error: str) -> HttpResponse:
        events.emit('oidc.login.failed', logging.INFO, error=error)
        if fail_url:
            return HttpResponseRedirect(_with_query(fail_url, {'error': error}))
        retry = escape(responses.login_url('/'))
        body = (
            '<!doctype html><html><head><meta charset="utf-8"><title>Sign-in failed</title></head>'
            '<body style="font-family:sans-serif;margin:2rem">'
            f'<p>Sign-in did not complete ({escape(error)}).</p>'
            f'<p><a href="{retry}">Try again</a></p></body></html>'
        )
        return HttpResponse(body, status=400, content_type='text/html; charset=utf-8')


class LogoutView(_OIDCView):
    """Sign out of this application only (the OP session stays)."""
    http_method_names = ['get', 'post']

    def get(self, request: HttpRequest) -> HttpResponse:
        return self.logout(request)

    def post(self, request: HttpRequest) -> HttpResponse:
        return self.logout(request)

    def _next(self, request: HttpRequest) -> Optional[str]:
        field = settings.OIDC_REDIRECT_OK_FIELD_NAME
        return safe_redirect_target(request, request.GET.get(field) or request.POST.get(field) or '/')

    def logout(self, request: HttpRequest) -> HttpResponse:
        next_url = self._next(request)
        if next_url is None:
            return _bad_redirect()
        row = _current_row(request)
        if row is not None:
            sessions.end_session(row, 'logout', delete_django_session=False)
        auth.logout(request)
        return HttpResponseRedirect(next_url)


class TotalLogoutView(LogoutView):
    """Sign out here and at the OP (RP-initiated logout with ``id_token_hint``)."""

    def logout(self, request: HttpRequest) -> HttpResponse:
        next_url = self._next(request)
        if next_url is None:
            return _bad_redirect()
        row = _current_row(request)
        id_token = None
        if row is not None:
            try:
                id_token = row.id_token
            except TokenDecryptError:
                id_token = None
            sessions.end_session(row, 'total_logout', delete_django_session=False)
        auth.logout(request)
        if not settings.OIDC_OP_TOTAL_LOGOUT:
            return HttpResponseRedirect(next_url)
        try:
            end_session_endpoint = get_provider().metadata().end_session_endpoint
        except (OIDCAuthUnavailable, OIDCConfigurationError):
            end_session_endpoint = None
        if not end_session_endpoint:
            return HttpResponseRedirect(next_url)
        state = get_random_string(max(32, settings.OIDC_RANDOM_SIZE))
        request.session[constants.SESSION_PENDING_LOGOUT] = {'state': state, 'next': next_url, 'ts': int(time.time())}
        params: Dict[str, Any] = {
            'client_id': settings.OIDC_RP_CLIENT_ID,
            'post_logout_redirect_uri': request.build_absolute_uri(reverse(constants.OIDC_URL_CALLBACK_NAME)),
            'state': state,
        }
        if id_token:
            params['id_token_hint'] = id_token
        return HttpResponseRedirect(_with_query(end_session_endpoint, params))


def _bcl_response(status: int = 200, error: Optional[str] = None) -> HttpResponse:
    response: HttpResponse = JsonResponse({'error': error}, status=status) if error else HttpResponse(status=status)
    response['Cache-Control'] = 'no-store'
    return response


@method_decorator(csrf_exempt, name='dispatch')
class BackchannelLogoutView(View):
    """OpenID Connect Back-Channel Logout 1.0: the OP POSTs a ``logout_token``."""
    http_method_names = ['post']

    def post(self, request: HttpRequest) -> HttpResponse:
        if not settings.OIDC_BACKCHANNEL_LOGOUT_ENABLED:
            return HttpResponseNotFound()
        token = request.POST.get('logout_token')
        if not token:
            return _bcl_response(400, 'invalid_request')
        try:
            claims = validate_logout_token(token)
        except OIDCTokenError as e:
            events.emit('oidc.backchannel_logout.rejected', logging.WARNING, reason=e.reason)
            return _bcl_response(400, 'invalid_request')
        except (OIDCAuthUnavailable, OIDCConfigurationError):
            events.emit('oidc.backchannel_logout.rejected', logging.WARNING, reason='keys_unavailable')
            return _bcl_response(503, 'temporarily_unavailable')
        issuer = str(claims.get('iss'))
        jti = claims.get('jti')
        if jti:
            cache_key = 'oidc:bcl:' + hashlib.sha256(f'{issuer}|{jti}'.encode('utf-8')).hexdigest()
            timeout = max(60, settings.OIDC_BACKCHANNEL_LOGOUT_MAX_AGE_SECONDS * 2)
            try:
                first = caches[settings.OIDC_CACHE_ALIAS].add(cache_key, 1, timeout=timeout)
            except Exception:  # the effect is idempotent anyway
                first = True
            if not first:
                events.emit('oidc.backchannel_logout.replay', logging.INFO)
                return _bcl_response(200)
        sid = str(claims.get('sid') or '') or None
        sub = str(claims.get('sub') or '') or None
        matched = sessions.end_sessions(issuer=issuer, sid=sid, sub=sub, reason='backchannel_logout')
        events.emit('oidc.backchannel_logout', logging.INFO, matched=matched, by='sid' if sid else 'sub')
        sessions._send(signals.oidc_backchannel_logout_received, issuer=issuer, sid=sid, sub=sub, matched=matched)
        return _bcl_response(200)


__all__ = [
    'AuthenticateView',
    'BackchannelLogoutView',
    'CallbackView',
    'LogoutView',
    'TotalLogoutView',
    'safe_redirect_target',
]
