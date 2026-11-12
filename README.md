django-oauth2-codeflow
======================

Long-lived, secure Django sign-ins with any OpenID Connect provider. Built and tested
against Keycloak 26.

- **Sign-in:** authorization code flow with PKCE (S256), `state` and `nonce` on every
  sign-in. Several sign-ins can be pending at once (one per browser tab).
- **Tokens:** kept in an encrypted database row per sign-in (`OIDCSession`), never in the
  Django session.
- **Refresh:** single-flight. Exactly one token call per sign-in, across threads and
  processes (`SELECT … FOR UPDATE` on the row).
- **Failures:** only `invalid_grant` ends a sign-in. When the provider is down or
  misconfigured, the answer is 503 and everybody stays signed in. A circuit breaker
  protects the provider.
- **HTTP contract:** one contract for browsers, frames and API clients (302, 401 JSON,
  503 JSON).
- **Logout:** OpenID Connect Back-Channel Logout, RP-initiated logout, and optional token
  revocation.
- **Bearer:** access tokens for machine clients. Only asymmetric signatures are accepted,
  and `iss`, `aud`/`azp` and `typ` are checked.
- **Integrations:** Django REST framework and Channels, plus test helpers and a pytest
  plugin.

Protocols: [OAuth 2.0](https://www.rfc-editor.org/rfc/rfc6749),
[PKCE](https://www.rfc-editor.org/rfc/rfc7636),
[OpenID Connect Core 1.0](https://openid.net/specs/openid-connect-core-1_0.html),
[RP-Initiated Logout 1.0](https://openid.net/specs/openid-connect-rpinitiated-1_0.html),
[Back-Channel Logout 1.0](https://openid.net/specs/openid-connect-backchannel-1_0.html),
[Token Revocation](https://www.rfc-editor.org/rfc/rfc7009).

Supported: Python 3.10 to 3.13; Django 4.2, 5.2 and 6.0.

**Upgrading from 2.x:** read [UPGRADING.md](UPGRADING.md) first. Sessions migrate by
themselves, but several defaults and answers changed.


Installation
------------

```bash
pip install 'django-oauth2-codeflow[drf,channels]'   # the extras are optional
```

This distribution provides the `oauth2_authcodeflow` package, the same name the upstream
`django-oauth2-authcodeflow` uses. Install only one of the two; system check
`oauth2_authcodeflow.E004` reports a conflict.


Setup
-----

```python
INSTALLED_APPS = [
    # ...
    'django.contrib.auth',
    'django.contrib.sessions',
    'oauth2_authcodeflow',
]

AUTHENTICATION_BACKENDS = [
    'oauth2_authcodeflow.auth.AuthenticationBackend',
    'django.contrib.auth.backends.ModelBackend',  # only if local accounts exist too
]

MIDDLEWARE = [
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'oauth2_authcodeflow.middleware.BearerAuthMiddleware',     # optional: machine clients
    'oauth2_authcodeflow.middleware.OIDCSessionMiddleware',
    'oauth2_authcodeflow.middleware.LoginRequiredMiddleware',  # optional: every URL needs a sign-in
    # ...
]

SESSION_COOKIE_AGE = 31 * 24 * 3600  # sliding; the provider decides when a sign-in ends

OIDC_OP_DISCOVERY_DOCUMENT_URL = 'https://sso.example.com/realms/acme/.well-known/openid-configuration'
OIDC_OP_ISSUER = 'https://sso.example.com/realms/acme'
OIDC_RP_CLIENT_ID = 'my-app'
OIDC_RP_CLIENT_SECRET = os.environ['OIDC_RP_CLIENT_SECRET']
OIDC_TOKEN_ENCRYPTION_KEYS = os.environ['OIDC_TOKEN_ENCRYPTION_KEYS'].split(',')  # Fernet keys, newest first
```

```python
# urls.py
urlpatterns = [
    path('oidc/', include('oauth2_authcodeflow.urls')),
    # ...
]
```

Then run `python manage.py migrate`, and `python manage.py oidc_check` to verify the
provider. `python manage.py oidc_urls app.example.com --secure` prints the URLs to register
on the client:

| Path | URL name | Purpose |
|---|---|---|
| `oidc/authenticate` | `oidc_authentication` | Starts a sign-in. Query: `next`, `fail`, `prompt` (`none`, `login`, `consent`, `select_account`), `login_hint`. |
| `oidc/callback` | `oidc_callback` | The redirect URI and the post-logout redirect URI. |
| `oidc/logout` | `oidc_logout` | Ends the local sign-in (GET or POST, `next`). |
| `oidc/total_logout` | `oidc_total_logout` | Also ends the SSO session at the provider (RP-initiated logout with `id_token_hint`). |
| `oidc/backchannel-logout` | `oidc_backchannel_logout` | Back-channel logout: the provider POSTs a `logout_token`. |

`next` and `fail` must stay on the request host or on `OIDC_ALLOWED_REDIRECT_HOSTS`;
anything else is refused with 400. When there is no `fail` URL, a failed sign-in renders a
small page with a retry link instead of redirecting.

The middleware order is checked (`oauth2_authcodeflow.E001`): Session, then Authentication,
then (Bearer), then OIDCSession, then LoginRequired.


How sign-ins work
-----------------

1. **Sign-in.** The callback validates the ID token (signature, `iss`, `aud`, `azp`, `exp`,
   `iat` and `nonce`, with 30 s of leeway). It provisions the user and creates an
   `OIDCSession` row. The Django session only holds the row's id.
   - The row keeps the encrypted access, refresh and ID tokens, `sub`, `sid`, the issuer,
     the expiry times, and the allow-listed profile claims (`OIDC_CLAIMS_ALLOWLIST`).
   - Signing in again in the same browser supersedes the previous row. The sign-ins of
     other devices are left alone.
2. **Every request.** `OIDCSessionMiddleware` loads the row and sets `request.oidc`.
   - In `eager` mode it refreshes when the access token has less than
     `OIDC_REFRESH_SKEW_SECONDS` left. In `lazy` mode it refreshes only when code asks for a
     token.
   - It slides the Django session at most once per `OIDC_SESSION_RENEW_INTERVAL_SECONDS`.
3. **Refresh.** A refresh is single-flight: a per-process lock, then a row lock.
   - A request whose token is still usable never waits for another refresher.
   - Other requests wait at most `OIDC_REFRESH_WAIT_SECONDS`.
   - When a refresh fails, the requests queued behind it get its outcome (the still-usable
     token, else 503) instead of each calling the provider again; the process retries
     that sign-in after `sessions.REFRESH_RETRY_GAP_SECONDS` (2).
   - A refresh never runs inside a caller's `transaction.atomic()` block.
4. **End.** A sign-in lasts as long as the provider's SSO session (Keycloak: *SSO Session
   Idle* / *SSO Session Max*). `OIDC_SESSION_MAX_AGE_SECONDS` can add an absolute cap.

What a refresh answer means:

| Token endpoint answer | Category | Effect |
|---|---|---|
| 200 with an access token | ok | New tokens, `oidc_session_refreshed` |
| `invalid_grant` | terminal | The sign-in ends (`refresh_invalid_grant`), and the request gets the "session expired" answer |
| `invalid_grant` "Invalid token issuer …" | configuration | Sign-in kept, 503, breaker open for 300 s |
| `invalid_client`, `unauthorized_client`, `unsupported_grant_type`, `invalid_scope`, `invalid_request` | configuration | Sign-in kept, 503, breaker open for 300 s |
| 429, 5xx, HTML pages, timeouts, connection errors, 200 without a token | transient | Sign-in kept; a still-usable token is served, otherwise 503 |

The circuit breaker opens after 5 failures in 30 s and stays open for 15 s. After that, one
probe call is let through. Configure it with `OIDC_CIRCUIT_BREAKER`. While it is open, no
request calls the provider.


The HTTP contract
-----------------

| Request kind | Recognised by | No sign-in | Sign-in ended | Provider unavailable |
|---|---|---|---|---|
| Navigation | GET with `Sec-Fetch-Mode: navigate`, or `Accept: text/html` without fetch metadata | 302 to `oidc/authenticate?next=…` | Same, after signing out | 503 HTML page that retries |
| Frame | `Sec-Fetch-Dest: iframe` (or `frame`, `embed`, `object`) | 401 HTML with a link that signs in in the top window | Same | 503 HTML |
| API | Anything else: fetch/XHR, `OIDC_API_URL_PATTERNS` (`^/api/`), non-GET, `Authorization` header | 401 `{"error": "not_authenticated", "login_url": …}` | 401 `{"error": "session_expired", "login_url": …}` | 503 `{"error": "auth_unavailable", "retry_after": n}` |

Every answer carries `Cache-Control: no-store`. A 401 carries `WWW-Authenticate`
(`Session realm="<client id>"`, or `Bearer error="invalid_token"` for a rejected bearer
token), and a 503 carries `Retry-After`. Anonymous requests never touch a pending sign-in.
Set `OIDC_NAVIGATION_REDIRECT = False` to answer navigations with 401 instead of a redirect.


Using the sign-in in code
-------------------------

```python
from oauth2_authcodeflow.tokens import get_access_token, get_id_claims

def view(request):
    token = get_access_token(request, min_ttl=60)  # refreshed if needed; None without an OIDC sign-in
    claims = get_id_claims(request)                # allow-listed profile claims
    roles = request.oidc.client_roles()            # resource_access[<client id>].roles
```

- `get_access_token` raises `OIDCSessionEnded` (answered 401) or `OIDCAuthUnavailable`
  (answered 503). The middleware maps both to the contract.
- Background code uses `oauth2_authcodeflow.sessions.access_token_for(session_id, min_ttl=0)`.
  Like a request, it returns a still-usable token at once while another thread refreshes;
  `wait=True` waits for that refresh, at most `OIDC_REFRESH_WAIT_SECONDS`.
- `oauth2_authcodeflow.utils.login_required` is Django's decorator, pointed at the OIDC
  sign-in.

**Signals** (`oauth2_authcodeflow.signals`):

| Signal | Arguments |
|---|---|
| `oidc_session_started` | `session`, `request` |
| `oidc_session_refreshed` | `session` |
| `oidc_session_refresh_failed` | `session`, `category`, `status`, `oauth_error` |
| `oidc_session_ended` | `session`, `reason` |
| `oidc_backchannel_logout_received` | `issuer`, `sid`, `sub`, `matched` |

**Events.** These go to the `oauth2_authcodeflow.events` logger as `event=<name> key=value`
lines, with tokens redacted: `oidc.session.started`, `.refreshed`, `.ended`,
`oidc.login.failed`, `oidc.backchannel_logout`, `oidc.circuit.opened` and `.closed`,
`oidc.config.issuer_mismatch`, `oidc.discovery.failed` and `.stale`.


Bearer tokens
-------------

`BearerAuthMiddleware` authenticates `Authorization: Bearer <access token>` without a
session.

- Only access tokens are accepted: `typ` must be `Bearer`, so an ID token is refused.
- The signature must be asymmetric (`OIDC_RP_SIGN_ALGOS_ALLOWED`); HS* and `none` are never
  accepted, whatever the setting says.
- `iss` must be `OIDC_OP_ISSUER`.
- `aud` must contain `OIDC_BEARER_AUDIENCES` (default: the client id). Alternatively, `azp`
  must be the client id when `OIDC_BEARER_ACCEPT_AZP = True`.

Other `Authorization` schemes are ignored.


Django REST framework
---------------------

```python
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'oauth2_authcodeflow.contrib.drf.OIDCBearerAuthentication',
        'oauth2_authcodeflow.contrib.drf.OIDCSessionAuthentication',
    ],
    'EXCEPTION_HANDLER': 'oauth2_authcodeflow.contrib.drf.oidc_exception_handler',
}
```

- Unauthenticated requests get 401 (not 403) with a `WWW-Authenticate` header.
- Error bodies carry a `code`: `not_authenticated`, `permission_denied` or `csrf_failed`.
- Library exceptions follow the contract: 401 `session_expired`, 503 `auth_unavailable`.


Channels
--------

```python
application = AuthMiddlewareStack(OIDCSessionASGIMiddleware(URLRouter(routes)))

class Consumer(OIDCSessionConsumerMixin, AsyncWebsocketConsumer):
    async def connect(self):
        if await self.oidc_connect():
            await self.accept()

    async def disconnect(self, code):
        await self.oidc_disconnect()
```

- Without a sign-in, the socket is accepted, then closed with code 4401.
- When the sign-in ends (logout, back-channel logout, `invalid_grant`), its sockets are
  closed with 4401 through a channel-layer group.
- Sockets also check the sign-in again every `oidc_revalidate_seconds` (300 by default).
- Synchronous consumers use `SyncOIDCSessionConsumerMixin`.


Keycloak client
---------------

| Client setting | Value |
|---|---|
| Client authentication | On (confidential). Public clients set `OIDC_RP_CLIENT_SECRET = ''` and rely on PKCE. |
| Valid redirect URIs | `https://app.example.com/oidc/callback` |
| Valid post logout redirect URIs (`post.logout.redirect.uris`) | `https://app.example.com/oidc/callback` |
| PKCE method (`pkce.code.challenge.method`) | `S256` |
| Back-channel logout URL (`backchannel.logout.url`) | `https://app.example.com/oidc/backchannel-logout`, reachable from Keycloak |
| Back-channel logout session required (`backchannel.logout.session.required`) | On |
| Front channel logout | Off |
| Audience mapper | Add the client id to access tokens, if bearer tokens are used |

Session policy lives in the realm, not in the app:

- *SSO Session Idle* and *SSO Session Max* decide how long people stay signed in. The
  reference policy is 30 days idle and 90 days max for application realms.
- Keep *Client Session Idle/Max* unset or at least as long.
- Keep *Revoke Refresh Token* off unless every client handles rotation.
- `SESSION_COOKIE_AGE` must be at least as long as the idle timeout: 31 days, sliding
  (warning `oauth2_authcodeflow.W003`).

**Internal URL.** The app can reach Keycloak through an internal URL (for example a cluster
Service) while users see the public one:

- Set `OIDC_OP_DISCOVERY_DOCUMENT_URL` to the internal URL.
- Set `OIDC_OP_ISSUER` to the public issuer.
- Keycloak must be started with `--hostname=https://sso.example.com` and
  `--hostname-backchannel-dynamic=true`. Its discovery document then keeps the public
  issuer while the token, JWKS and userinfo endpoints use the internal URL.
- Without these options, Keycloak answers refreshes with `invalid_grant` "Invalid token
  issuer". The library treats that as a configuration error (503, nobody signed out).
- Error `oauth2_authcodeflow.E003` flags an internal discovery host without
  `OIDC_OP_ISSUER`.


Settings
--------

Only `OIDC_RP_CLIENT_ID`, `OIDC_RP_CLIENT_SECRET` and the provider location are required.

| Setting | Default | Meaning |
|---|---|---|
| `OIDC_OP_DISCOVERY_DOCUMENT_URL` | — | Discovery document. The app may use an internal URL. |
| `OIDC_OP_ISSUER` | discovery `issuer` | Public issuer. Must equal the discovery document's `issuer` (a mismatch is a configuration error). |
| `OIDC_OP_AUTHORIZATION_URL`, `_TOKEN_URL`, `_USERINFO_URL`, `_JWKS_URL`, `_END_SESSION_URL`, `_REVOCATION_URL` | discovery | Explicit endpoints, applied on top of discovery. |
| `OIDC_OP_FETCH_USER_INFO` | `True` | Merge userinfo claims at sign-in (when `sub` matches). |
| `OIDC_OP_TOTAL_LOGOUT` | `True` | `total_logout` also ends the SSO session. |
| `OIDC_RP_CLIENT_AUTH_METHOD` | `client_secret_post`, or `none` without a secret | `client_secret_post`, `client_secret_basic` or `none`. |
| `OIDC_RP_USE_PKCE` | `True` | PKCE S256. |
| `OIDC_RP_SCOPES` | `['openid', 'email', 'profile']` | Requested scopes. |
| `OIDC_RP_SIGN_ALGOS_ALLOWED` | `['RS256', 'PS256', 'ES256']` | Accepted signature algorithms. Symmetric ones are refused (E002). |
| `OIDC_RP_IDP_SIGN_KEY` | `None` | Static public key (PEM) instead of JWKS. |
| `OIDC_ALLOWED_REDIRECT_HOSTS` | `[]` | Extra hosts allowed for `next` and `fail`. |
| `OIDC_LOGIN_STATE_TTL_SECONDS` / `OIDC_LOGIN_MAX_PENDING` | `600` / `5` | Pending sign-ins per browser. |
| `OIDC_CREATE_USER` | `True` | Create unknown users at sign-in. |
| `OIDC_UNUSABLE_PASSWORD` | `False` | Make the password unusable at every sign-in. New users always get an unusable password. |
| `OIDC_DJANGO_USERNAME_FUNC`, `OIDC_EMAIL_CLAIM`, `OIDC_FIRSTNAME_CLAIM`, `OIDC_LASTNAME_CLAIM`, `OIDC_EXTEND_USER` | as in 2.x | User provisioning hooks. |
| `OIDC_CLAIMS_ALLOWLIST` | profile claims | Claims kept on the row (`'*'` keeps all; tokens are never kept here). |
| `OIDC_HTTP_TIMEOUT` | `(3.05, 8)` | Connect and read timeout of every call to the provider. |
| `OIDC_CA_BUNDLE` / `OIDC_PROXY` | `True` / `None` | TLS verification and proxies for provider calls. |
| `OIDC_REFRESH_MODE` | `'eager'` | `eager` refreshes in the middleware; `lazy` refreshes only in `get_access_token`. |
| `OIDC_REFRESH_SKEW_SECONDS` | `60` | Refresh when less than this is left. |
| `OIDC_REFRESH_WAIT_SECONDS` | `10` | Longest wait for another refresher (and the Postgres `lock_timeout`). |
| `OIDC_CIRCUIT_BREAKER` | `{'failures': 5, 'window': 30, 'cooldown': 15, 'config_cooldown': 300}` | Per-process breaker in front of the token endpoint. |
| `OIDC_SESSION_RENEW_INTERVAL_SECONDS` | `3600` | How often the Django session slides. |
| `OIDC_SESSION_MAX_AGE_SECONDS` | `None` | Optional absolute cap on a sign-in. |
| `OIDC_TOKEN_ENCRYPTION_KEYS` | derived from `SECRET_KEY` | Fernet keys, newest first. Without them, keys are derived from `SECRET_KEY` and `SECRET_KEY_FALLBACKS` (warning W001 outside DEBUG). The derived keys always stay as decrypt-only fallbacks, so setting explicit keys later keeps every sign-in; `oidc_rotate_keys` then moves the tokens onto the first key. |
| `OIDC_LEGACY_SESSION_MIGRATION` | `True` | Move 2.x session tokens into a row on their first request. |
| `OIDC_LOGIN_REQUIRED_EXEMPT_URL_PATTERNS` | `[]` | Regexes `LoginRequiredMiddleware` lets through. Anchor them (W002). |
| `OIDC_SESSION_EXEMPT_URL_PATTERNS` | `[]` | Regexes `OIDCSessionMiddleware` skips (health checks). |
| `OIDC_API_URL_PATTERNS` | `['^/api/']` | Paths that always get API answers. |
| `OIDC_NAVIGATION_REDIRECT` | `True` | Redirect unauthenticated navigations to the sign-in. |
| `OIDC_BEARER_AUDIENCES` / `OIDC_BEARER_ACCEPT_AZP` | client id / `False` | Bearer audience rules. |
| `OIDC_REVOKE_ON_LOGOUT` | `False` | Revoke the refresh token on `logout` and `total_logout`. |
| `OIDC_BACKCHANNEL_LOGOUT_ENABLED` | `True` | Serve the back-channel logout endpoint. |
| `OIDC_BACKCHANNEL_LOGOUT_MAX_AGE_SECONDS` | `300` | Accepted age of a `logout_token`. |
| `OIDC_CACHE_ALIAS` | `'default'` | Cache for the `logout_token` replay check (`jti`). |
| `OIDC_CHANNELS_NOTIFY` | `'auto'` | `off` disables socket notifications. |

Renamed and removed settings are listed in [UPGRADING.md](UPGRADING.md). Old names keep
working, with a `DeprecationWarning`.


Operations
----------

**Commands:**

| Command | Purpose |
|---|---|
| `oidc_check [--json]` | Checks discovery, issuer, JWKS and token endpoint reachability. |
| `oidc_urls <host> [--secure]` | Prints the redirect, post-logout and back-channel URLs. |
| `oidc_cleanup [--ended-older-than DAYS] [--dry-run]` | Ends expired and orphaned rows, deletes old ended rows, and purges the 2.x blacklist. Run it daily. |
| `oidc_rotate_keys [--batch-size N] [--end-undecryptable]` | Re-encrypts every row with the newest key, after adding one to `OIDC_TOKEN_ENCRYPTION_KEYS` (also the first time, when the tokens were written with the derived key). |

**System checks:**

| Id | Check |
|---|---|
| E001 | Middleware order |
| E002 | Symmetric algorithms configured |
| E003 | Internal discovery host without `OIDC_OP_ISSUER` |
| E004 | The upstream distribution is also installed |
| W001 | Derived encryption keys outside DEBUG |
| W002 | Unanchored exempt patterns |
| W003 | `SESSION_COOKIE_AGE` under 30 days |

**Admin.** Sign-ins are read-only in the admin (no tokens shown), with an action that ends
the selected ones.


Testing your project
--------------------

```python
from oauth2_authcodeflow.testing import create_oidc_session

def test_dashboard(client, django_user_model):
    user = django_user_model.objects.create_user('alice')
    create_oidc_session(client, user, client_roles=['admin'])
    assert client.get('/dashboard/').status_code == 200
```

- `create_oidc_session(client_or_request, user, *, client_roles=None, claims=None,
  access_expires_in=300, refresh=True, **access_claims)` signs a user in without a
  provider. It works with a test `Client` or an `HttpRequest`.
- The pytest fixture `oidc_op` (loaded through the `pytest11` entry point) installs
  `FakeOP`, an in-process provider. It covers discovery, JWKS, token, userinfo and
  revocation, and the token endpoint's answers can be scripted:
  `oidc_op.script_token((400, {'error': 'invalid_grant'}))`.
- `make_signed_token(claims)` signs tokens with the key `FakeOP` publishes.


Developing this library
-----------------------

```bash
poetry install --with dev,typing --all-extras
poetry run pytest test                                                    # unit tests (SQLite)
PGHOST=/tmp PGUSER=postgres poetry run pytest test/pg -m postgres --ds=test.pg.settings_pg
poetry run pytest test/keycloak -m keycloak --ds=test.keycloak.settings_kc  # starts Keycloak 26 in Docker
poetry run flake8 oauth2_authcodeflow test && poetry run mypy oauth2_authcodeflow
```

The Postgres suite exercises row locking, which SQLite ignores. The Keycloak suite starts
the `kc-codeflow-3` container on 127.0.0.1:18080 and removes it afterwards.
