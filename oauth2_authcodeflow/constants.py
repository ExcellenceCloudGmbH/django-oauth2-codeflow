# Session keys of the 2.x session layout. 3.0 keeps tokens in ``OIDCSession`` rows and only
# reads these keys to migrate a legacy session (see ``sessions.migrate_legacy``).
SESSION_OP_AUTHORIZATION_URL = 'oidc_op_authorization_url'
SESSION_OP_TOKEN_URL = 'oidc_op_token_url'
SESSION_OP_USERINFO_URL = 'oidc_op_userinfo_url'
SESSION_OP_JWKS_URL = 'oidc_op_jwks_url'
SESSION_OP_JWKS = 'oidc_op_jwks'
SESSION_OP_END_SESSION_URL = 'oidc_op_end_session_url'
SESSION_OP_CLAIMS_PARAMETER_SUPPORTED = 'oidc_op_claims_parameter_supported'
SESSION_NEXT_URL = 'oidc_next_url'
SESSION_FAIL_URL = 'oidc_fail_url'
SESSION_STATE = 'oidc_state'
SESSION_NONCE = 'oidc_nonce'
SESSION_CHALLENGE = 'oidc_challenge'
SESSION_ID_TOKEN = 'oidc_id_token'
SESSION_ACCESS_TOKEN = 'oidc_access_token'
SESSION_ACCESS_EXPIRES_AT = 'oidc_access_expires_at'
SESSION_REFRESH_TOKEN = 'oidc_refresh_token'
SESSION_EXPIRES_AT = 'oidc_expires_at'
SESSION_LOGOUT_STATE = 'oidc_logout_state'

LEGACY_SESSION_KEYS = (
    SESSION_OP_AUTHORIZATION_URL,
    SESSION_OP_TOKEN_URL,
    SESSION_OP_USERINFO_URL,
    SESSION_OP_JWKS_URL,
    SESSION_OP_JWKS,
    SESSION_OP_END_SESSION_URL,
    SESSION_OP_CLAIMS_PARAMETER_SUPPORTED,
    SESSION_NEXT_URL,
    SESSION_FAIL_URL,
    SESSION_STATE,
    SESSION_NONCE,
    SESSION_CHALLENGE,
    SESSION_ID_TOKEN,
    SESSION_ACCESS_TOKEN,
    SESSION_ACCESS_EXPIRES_AT,
    SESSION_REFRESH_TOKEN,
    SESSION_EXPIRES_AT,
    SESSION_LOGOUT_STATE,
)

# 3.0 session layout (besides OIDC_SESSION_KEY, a setting)
SESSION_PENDING_LOGINS = '_oidc_pending'
SESSION_PENDING_LOGOUT = '_oidc_pending_logout'
SESSION_RENEWED_AT = '_oidc_renewed'

OIDC_URL_AUTHENTICATION_NAME = 'oidc_authentication'
OIDC_URL_CALLBACK_NAME = 'oidc_callback'
OIDC_URL_LOGOUT_NAME = 'oidc_logout'
OIDC_URL_TOTAL_LOGOUT_NAME = 'oidc_total_logout'
OIDC_URL_BACKCHANNEL_LOGOUT_NAME = 'oidc_backchannel_logout'
# Removed in 3.0 (front-channel logout by the OP); kept for imports only.
OIDC_URL_LOGOUT_BY_OP_NAME = 'oidc_logout_by_op'
OIDC_FROM_CLI_QUERY_STRING = 'from_cli'
OIDC_CLAIMS_PARAMETER_SUPPORTED = 'claims_parameter_supported'

BACKCHANNEL_LOGOUT_EVENT = 'http://schemas.openid.net/event/backchannel-logout'

AUTH_BACKEND_PATH = 'oauth2_authcodeflow.auth.AuthenticationBackend'
