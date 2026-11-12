Upgrading
=========

2.x to 3.0
----------

3.0 makes three changes:

- The tokens move out of the Django session into an encrypted `OIDCSession` row per
  sign-in.
- A refresh is single-flight, and only `invalid_grant` ends a sign-in.
- Browsers, frames and API clients get one HTTP contract (see the README).

Signed-in users stay signed in across the upgrade: a 2.x session moves into a row on its
first 3.0 request.

### Steps

1. **Install** `django-oauth2-codeflow==3.0.0`, with the `drf` or `channels` extra if you
   use them.
   - Remove the upstream `django-oauth2-authcodeflow` if it is installed. Both provide the
     `oauth2_authcodeflow` package (check E004).
2. **Migrate:** `python manage.py migrate oauth2_authcodeflow`. Migration `0006` creates the
   `OIDCSession` table.
   - The app config pins `default_auto_field = 'django.db.models.AutoField'`. Projects with
     `DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'` therefore get no spurious
     migration for this app.
   - Never commit a migration that `makemigrations` generates for `oauth2_authcodeflow` in
     your project. Delete it.
3. **Settings.**
   - Rename the settings listed below. The old names still work and warn.
   - Set `OIDC_OP_ISSUER` to the public issuer.
   - Set `OIDC_TOKEN_ENCRYPTION_KEYS` (Fernet keys from a secret store, newest first).
   - Set `SESSION_COOKIE_AGE` to at least the provider's idle timeout (31 days). The session
     slides by itself.
   - Remove `OIDC_MIDDLEWARE_SESSION_TIMEOUT_SECONDS`.
4. **Middleware.** Use the order of the README: `BearerAuthMiddleware` (optional),
   `OIDCSessionMiddleware`, `LoginRequiredMiddleware`, all after `AuthenticationMiddleware`.
   - `RefreshAccessTokenMiddleware` and `RefreshSessionMiddleware` are deprecated aliases of
     `OIDCSessionMiddleware`. A request still runs it only once.
5. **Provider client** (Keycloak):
   - Add `oidc/callback` to the valid post-logout redirect URIs.
   - Set the back-channel logout URL to `https://<app>/oidc/backchannel-logout`, with
     "session required" on.
   - To use an internal URL, configure the hostname options described in the README first.
6. **Code.**
   - Replace reads of `request.session['oidc_access_token']` (and the other `oidc_*` keys)
     with `oauth2_authcodeflow.tokens.get_access_token(request)` and `get_id_claims(request)`,
     or with `request.oidc`.
   - In tests, replace hand-made `session['oidc_*']` values with
     `oauth2_authcodeflow.testing.create_oidc_session(client, user)`.
7. **Clients.** Frontends must follow the contract:
   - 401 `{"error": "session_expired" | "not_authenticated", "login_url"}`: sign in again
     through `login_url`.
   - 503 `{"error": "auth_unavailable", "retry_after"}`: retry later. It is not a logout.
   - A 401 or 403 must never trigger a logout at the provider.
8. **Operations.**
   - Run `python manage.py oidc_cleanup` daily.
   - Run `python manage.py oidc_check` after deploying.

### Settings

| 2.x | 3.0 |
|---|---|
| `OIDC_TIMEOUT` | `OIDC_HTTP_TIMEOUT`, now applied to every call to the provider (default `(3.05, 8)`) |
| `OIDC_RP_FORCE_SECRET_WITH_PKCE = True` | `OIDC_RP_CLIENT_AUTH_METHOD = 'client_secret_post'` |
| `OIDC_MIDDLEWARE_NO_AUTH_URL_PATTERNS` | `OIDC_LOGIN_REQUIRED_EXEMPT_URL_PATTERNS` |
| `OIDC_MIDDLEWARE_NO_REFRESH_URL_PATTERNS` | `OIDC_SESSION_EXEMPT_URL_PATTERNS` |
| `OIDC_MIDDLEWARE_API_URL_PATTERNS` | `OIDC_API_URL_PATTERNS` (default `['^/api/']`) |
| `OIDC_MIDDLEWARE_LOGIN_REQUIRED_REDIRECT` | `OIDC_NAVIGATION_REDIRECT` |
| `OIDC_MIDDLEWARE_SESSION_TIMEOUT_SECONDS` (7-day cap) | Removed. The provider's SSO session decides; `OIDC_SESSION_MAX_AGE_SECONDS` adds an optional cap. |
| `OIDC_VIEW_LOGOUT_BY_OP` | Removed. Use back-channel logout (`OIDC_VIEW_BACKCHANNEL_LOGOUT`). |

### Behaviour changes

- **Passwords.** `OIDC_UNUSABLE_PASSWORD` now defaults to False. New users still get an
  unusable password, but an existing user's password is no longer reset at each sign-in.
  In 2.x, that reset signed the user out on every other device.
- **Inactive users stay inactive.** The sign-in is refused (`user_inactive`); 2.x set
  `is_active = True` at every sign-in.
- **Signing algorithms.** Symmetric (HS\*) and `none` signatures are never accepted, even
  if `OIDC_RP_SIGN_ALGOS_ALLOWED` lists them (E002). The default list is RS256, PS256 and
  ES256.
- **Bearer tokens** must be access tokens:
  - `typ` must be `Bearer`, so ID tokens are refused.
  - `iss` must match, and so must `aud` (or `azp`, if you enable it).
  - Discovery and JWKS are cached per process, and no session row is written per request.
- **Answers.** API requests get 401 JSON instead of a 302 redirect or a 403. Anonymous
  fetch/XHR is never redirected. With `contrib.drf`, DRF answers 401 with a
  `WWW-Authenticate` header.
- **Pending sign-ins.** They are keyed by `state`, so several tabs can sign in at once. A
  callback with `?error=` (for example after `prompt=none`) no longer signs the current
  user out.
- **Refresh failures.** Only `invalid_grant` ends a sign-in. Provider outages,
  configuration errors and issuer mismatches answer 503 and keep the sign-in.
- **Session lifetime.** The Django session slides with `SESSION_COOKIE_AGE`. The fixed
  7-day cap is gone.
- **Revocation.** The refresh token is revoked at logout only with
  `OIDC_REVOKE_ON_LOGOUT = True`.
- **`oidc_urls`** prints the back-channel logout URL instead of the logout-by-OP URL.

### Removed

- `LogoutByOPView` and its URL (front-channel "logout by OP"). Use back-channel logout.
- The `from_cli` sign-in parameter (`/FROM_CLI_OK`).
- HS\* validation of ID tokens and bearer tokens.
- Tokens in the Django session. The 2.x keys are migrated once, then deleted.

### Deprecated (removed in 4.0)

- `RefreshAccessTokenMiddleware`, `RefreshSessionMiddleware` and `Oauth2MiddlewareMixin`.
- `MiddlewareException`; use `OIDCLoginRequired`.
- The 2.x setting names listed above.
- `purge_blacklisted_tokens` (use `oidc_cleanup`) and the `BlacklistedToken` model, which is
  frozen.

### Rolling back to 2.1.0

- 2.1.0 does not read `OIDCSession` rows. Users whose session was migrated sign in again
  once; the provider's SSO cookie makes that a redirect round trip, without a password
  prompt.
- The table can stay. `python manage.py migrate oauth2_authcodeflow 0005` removes it.
- If you use an internal discovery URL, unset it before rolling back. 2.1.0 does not
  separate the issuer from the discovery URL.
- Remove the back-channel logout URL from the provider client if 2.1.0 does not serve it.
