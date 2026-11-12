# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## 3.0.0
### Security
- Tokens are no longer stored in the Django session. They live in an `OIDCSession` row per sign-in, encrypted at rest (Fernet; `OIDC_TOKEN_ENCRYPTION_KEYS`, or keys derived from `SECRET_KEY` and `SECRET_KEY_FALLBACKS`, which stay as decrypt-only fallbacks so explicit keys can be adopted without signing anyone out; `oidc_rotate_keys`).
- Symmetric (HS\*) and `none` signatures are never accepted. The default algorithms are RS256, PS256 and ES256.
- Bearer authentication accepts only access tokens (`typ` Bearer), with `iss` and `aud`/`azp` checked. ID tokens are refused.
- PKCE (S256), `state` and `nonce` on every sign-in. Pending sign-ins are kept per `state` with a TTL. `next` and `fail` are restricted to allowed hosts (400 otherwise).
- Calls to the provider never send or store cookies, never follow redirects and always time out (`OIDC_HTTP_TIMEOUT`).
- Logs and responses never contain tokens; JWTs are redacted from logged values.
### Fixed
- A provider outage, an HTML error page, a timeout or a configuration error no longer signs users out. Only `invalid_grant` ends a sign-in; everything else answers 503 and keeps it.
- Concurrent requests no longer race on refresh. The refresh is single-flight across threads and processes (in-process lock plus `SELECT … FOR UPDATE`), and lost updates of rotated tokens are gone. A request whose token is still usable never waits for it, and the requests queued behind a failed refresh get its outcome instead of each calling the provider again. On PostgreSQL the row lock is `FOR NO KEY UPDATE`, so rows that reference a sign-in can be written while it refreshes.
- Anonymous fetch/XHR requests get 401 JSON instead of a redirect that fails CORS, and they never destroy a pending sign-in.
- A callback `?error=` (for example after `prompt=none`) no longer signs the current user out. Several tabs can sign in at once.
- Signing in no longer resets the password of existing users, which signed them out on every other device.
- Inactive Django users are no longer re-activated at sign-in.
- JWKS are cached per process and refetched once on an unknown `kid`. Discovery is cached, and a stale copy is used while the provider is down.
- `OIDC_TIMEOUT`, `OIDC_PROXY` and `OIDC_CREATE_USER` are applied.
### Changed
- New middlewares: `OIDCSessionMiddleware` (refresh and sliding session), `LoginRequiredMiddleware`, `BearerAuthMiddleware`. All of them answer with one contract: 302 for navigations, 401 HTML for frames, 401 JSON `not_authenticated`/`session_expired` for API requests, and 503 JSON `auth_unavailable` with `Retry-After`. Every answer is `no-store`.
- How long a sign-in lasts is decided by the provider's SSO session. The Django session slides with `SESSION_COOKIE_AGE`; `OIDC_SESSION_MAX_AGE_SECONDS` adds an optional cap.
- `OIDC_UNUSABLE_PASSWORD` defaults to False and is applied to new users only.
- Settings renamed (old names still work and warn): `OIDC_TIMEOUT` → `OIDC_HTTP_TIMEOUT`, `OIDC_RP_FORCE_SECRET_WITH_PKCE` → `OIDC_RP_CLIENT_AUTH_METHOD`, `OIDC_MIDDLEWARE_NO_AUTH_URL_PATTERNS` → `OIDC_LOGIN_REQUIRED_EXEMPT_URL_PATTERNS`, `OIDC_MIDDLEWARE_NO_REFRESH_URL_PATTERNS` → `OIDC_SESSION_EXEMPT_URL_PATTERNS`, `OIDC_MIDDLEWARE_API_URL_PATTERNS` → `OIDC_API_URL_PATTERNS`, `OIDC_MIDDLEWARE_LOGIN_REQUIRED_REDIRECT` → `OIDC_NAVIGATION_REDIRECT`.
- `oidc_urls` prints the redirect, post-logout and back-channel logout URLs. It warns when the host is not in `ALLOWED_HOSTS`.
- Supported versions: Django 4.2, 5.2 and 6.0; Python 3.10 to 3.13.
### Added
- `OIDCSession` model (migration `0006`). A 2.x session moves into a row on its first request (`OIDC_LEGACY_SESSION_MIGRATION`).
- OpenID Connect Back-Channel Logout 1.0 at `oidc/backchannel-logout`. It validates the `logout_token`, ends sign-ins by `sid` or `sub`, and checks replays per `jti`. RP-initiated logout sends `id_token_hint`. Refresh-token revocation is optional (`OIDC_REVOKE_ON_LOGOUT`).
- Separate public issuer (`OIDC_OP_ISSUER`) and internal discovery URL.
- A token-endpoint error taxonomy and a per-process circuit breaker (`OIDC_CIRCUIT_BREAKER`).
- `request.oidc`, `get_access_token(request, min_ttl=)`, `get_id_claims(request)`, `sessions.access_token_for(session_id, min_ttl=, wait=)`, and `client_roles()`.
- Signals: `oidc_session_started`, `oidc_session_refreshed`, `oidc_session_refresh_failed`, `oidc_session_ended`, `oidc_backchannel_logout_received`. Structured events go to the `oauth2_authcodeflow.events` logger.
- `contrib.drf`: session and bearer authentication with `authenticate_header` (401, not 403), and an exception handler with error codes.
- `contrib.channels`: ASGI middleware and consumer mixins. Sockets close with 4401 when their sign-in ends.
- Commands `oidc_check`, `oidc_cleanup` and `oidc_rotate_keys`; a read-only admin for sign-ins.
- System checks E001 to E004 and W001 to W003.
- `oauth2_authcodeflow.testing`: `create_oidc_session`, `FakeOP`, `make_signed_token`, and the pytest fixture `oidc_op` (entry point `pytest11`).
- CI (GitHub Actions): flake8 and mypy; a unit matrix of Python 3.10–3.13 × Django 4.2/5.2/6.0; a Postgres 16 concurrency suite; a nightly Keycloak 26 integration suite.
### Removed
- `LogoutByOPView` and `OIDC_VIEW_LOGOUT_BY_OP`; use back-channel logout.
- `OIDC_MIDDLEWARE_SESSION_TIMEOUT_SECONDS`, the fixed 7-day cap.
- The `from_cli` sign-in parameter.
- Tokens in the Django session (`oidc_*` keys).
### Deprecated
- `RefreshAccessTokenMiddleware`, `RefreshSessionMiddleware`, `Oauth2MiddlewareMixin` and `MiddlewareException`.
- `purge_blacklisted_tokens` (use `oidc_cleanup`). `BlacklistedToken` is frozen and will be dropped in 4.0.
## 2.1.0
### Fixed
- A token refresh only ends the session when the provider answers `invalid_grant`. Other OAuth errors, 429/5xx, HTML pages from a proxy, timeouts and connection errors keep the session: the still valid access token is served, otherwise a 503 with `Retry-After` is answered (`{"error": "auth_unavailable", "retry_after": N}` for API/fetch requests, a self-retrying page for navigations).
- The id token is no longer blacklisted on refresh nor checked against the blacklist on every request: concurrent requests holding the pre-refresh session no longer log the user out (the cause of the `SessionInterrupted` errors).
- The refresh lock has an owner and is only released by it; concurrent requests use the current token while it is valid and otherwise wait (read only) for the leader instead of refreshing in parallel; an unusable lock cache falls back to a process-local lock instead of a refresh stampede. New setting `OIDC_REFRESH_LOCK_CACHE_ALIAS`.
- `oidc_expires_at` slides on every successful refresh (idle timeout instead of a fixed lifetime).
- Fetch/XHR/API requests get a JSON 401 (`{"error": "not_authenticated" | "session_expired", "login_url": …}`, `WWW-Authenticate`, `Cache-Control: no-store`) instead of a redirect, also for GET requests on API URLs; page navigations keep the redirect. Anonymous sessions are never flushed (a pending login survives concurrent XHRs).
- A callback carrying an authorization code is never handled as a logout callback.
### Changed
- The access token is refreshed `OIDC_REFRESH_SKEW_SECONDS` (30) before it expires. New setting `OIDC_RETRY_AFTER_SECONDS` (5).
- With PKCE, `state` and `nonce` are sent and validated too.
- A session written by the 3.0 layout (`_oidc_session_id` without tokens) is treated as unauthenticated without being destroyed (safe rollback from 3.0).
- `BearerAuthMiddleware` ignores `Authorization` headers using another scheme than `OIDC_AUTHORIZATION_HEADER_PREFIX`.
- Every forced logout is logged at WARNING with its reason; tokens never appear in messages, logs or response bodies.
### Security
- `OIDC_RP_SIGN_ALGOS_ALLOWED` defaults to RS/PS/ES 256/384/512: HS* (keyed by the client secret) must be listed explicitly. PS* and ES* keys from the JWKS are supported.
- `OIDC_UNUSABLE_PASSWORD` defaults to `False`: logging in no longer changes the password hash, which invalidated the user's other sessions.
- `OIDC_TIMEOUT` defaults to 10 seconds and, like `OIDC_PROXY`, is applied to every call to the provider (discovery, JWKS, token, refresh, userinfo).

## 2.0.0
### Security
- Replace `python-jose` by `PyJWT[crypto]>=2.15`: `python-jose` <= 3.5.0 (GHSA-3qf3-8w2g-rqmx) and its `ecdsa` dependency (GHSA-wj6h-64fc-37mp) have no fixed release. The ID token validation keeps the previous rules (signature, `exp`, `nbf`, `jti`, `aud` only when present, nonce).
- Security updates of the locked dependencies (Django 5.2.18, cryptography 50.0.2, urllib3 2.8.0, requests 2.34.2, idna 3.20, sqlparse 0.6.0, Jinja2 3.1.6, Pygments 2.21.0, pytest 9.1.1, certifi 2026.10.19 for GHSA-6q48-6c7w-9gw2)
### Changed
- Python 3.13, Django 5.2 compat
### Removed
- Python 3.8 and 3.9 support (the patched Django, cryptography, urllib3, requests, sqlparse and pytest releases require Python 3.10+)

## 1.2.3
### Changed
- Security update for `python-jose` from version `3.3.0` to `3.4.0`

## 1.2.2
### Fixed
- 'Origin' header should NOT be present if the Azure app is not a SPA.
### Added
- Explicit compatibility with django 5.1

## 1.2.1
### Fixed
- Fix a migration error from version `1.1.0` on a non-empty database (gitlab #26).
### Added
- Explicit compatibility with django 5.1

## 1.2.0
### Security
- Security package upgrades
### Fixed
- Azure tenant PKCE public app fix (Origin header was missing)
- Final fix for mysql on InnoDB with max key of 3072 by having the constraint as a lonely migration (github #21). Thanks Jurymax99 for the suggested merge request.
- Allow to logout even when using the Django `ModelBackend` (github #25)
### Changed
- Do not send the client secret, even if defined, with `PKCE` by default (github #18)
  This can be overriden with the `OIDC_RP_FORCE_SECRET_WITH_PKCE` parameter.
- Gitlab CI upgrades

## 1.1.0
### Fixed
- redirect after total logout could happen with a GET (#10)
- allow empty client secret (QE-625, gitlab #9)
### Added
- User logged in signal doc example, thanks @pinoatrome (github #16)
- Drop python 3.7, support python 3.12 and django 5

## 1.0.1
### Fixed
- Fix timestamp-awareness inside `RefreshSession` and `RefreshAccessToken` middlewares

## 1.0.0
### Changed
- Each log (debug, warning, error) is now correctly bound to the module name.
- Mypy 1.0
### Added
- Added documentation and changelog urls for PyPI 

## 0.9.0
### Fixed
- Default value for `jwks` in `BearerAuthenticationBackend` should be dict, not a list.
- Fix blacklist expiration for token where seconds where used as hours
- Fix `_clear_cache` method in `CacheBaseView`: was not clearing the session correctly.
- Configuration cannot be updated when using unit tests. This is now fixed. No impact on lib usage.
- Respect the optional `fail` parameter of `@login_required` decorator.
- Middlewares should not inherit depraceted `MiddlewareMixin`.
- If user does not exist on request, should not crash in `Oauth2MiddlewareMixin.is_oidc_enabled`.
### Changed
- Allow to override `MIN_SECONDS` in `RefreshSessionMiddleware`.
- Use UTC time in `RefreshAccessTokenMiddleware`, `RefreshSessionMiddleware`.
### Added
- `LoginRequiredMiddleware`
- Documentation about `@login_required`
### Removed
- `pytz` removed. `datetime.timezone.utc` is the only thing required.

## 0.8.1
### Fixed
- urls listed in `OIDC_MIDDLEWARE_NO_AUTH_URL_PATTERNS` will not be tried on authentication in `auth.py`

## 0.8.0
### Added
- Allow to specify `userinfo` and `id_token` individual claims to get along with the id token request if the OP supports it (Eric Plaster, mr !12).
### Changed
- `OIDC_EXTEND_USER` callable can now takes a `request` and `access_token` as additional arguments (compatibility is assured).
- Migrate can raise an `IntegrityError` (ticket #7).
- All parameters that accept a function can also accept a dotted string to import the function.
- Migrate from `pipenv` to `poetry` system.

## 0.7.0
### Added
- Missing Django migration
### Changed
- Allow Django 4.1+ (but not 5.0)
- Add Python 3.11 in classifier
- Dependencies upgrade

## 0.6.0
### Changed
- Allow usage with Django 4.0 and update classifiers
- Make the code compatible with Python 3.7

## 0.5.0
### Added
- Allow to scramble the password only when creating an account instead of each SSO connection/renewal

## 0.4.0
### Added
- Allow `user` extension with a callable using `claims`
### Fixed
- User `email` field was filled with raw `email` value instead of actual value if `OIDC_EMAIL_CLAIM` was not set.

## 0.3.2
### Fixed
- No error 500 on expired authentication because the database session might not be found

## 0.3.1
### Fixed
- Prevent infinite redirect to authenticate view when using any middleware (session was not cleared properly)

## 0.3.0
### Changed
- Use `Authorization` header for `USERINFO` instead of request param
- `token` field in `BlacklistedToken` table changed from `TextField` to `CharField(max_length=15000)` for MySql compatibility
### Fixed
- register json web keys to session only if not already registered
- fix error handling by adding required method parameter
- `email`, `first_name` and `last_name` cannot be None. Fallback to empty string.
- correctly check for status code ok when getting access token.

## 0.2.1
### Fixed
- fix doc about `SESSION_COOKIE_SECURE`
- fix typo in f-string

## 0.2.0
### Added
- OP.md with settings examples for multiple OIDC Providers
### Fixed
- Management commands were not included in the package

## 0.1.0
Initialize library
