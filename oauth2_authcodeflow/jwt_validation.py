"""Validation of id tokens, access tokens (bearer) and back-channel logout tokens.

Only asymmetric algorithms are accepted — whatever ``OIDC_RP_SIGN_ALGOS_ALLOWED`` says,
``HS*`` and ``none`` never pass (a client secret is shared with every application of the
client and must not be able to mint tokens). Errors are ``OIDCTokenError`` with a short
reason; the token itself is never part of an error.
"""
import time
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

import jwt
from jwt.exceptions import PyJWTError
from jwt.types import Options

from . import constants
from .conf import settings
from .exceptions import OIDCTokenError
from .provider import get_provider

ASYMMETRIC_ALGORITHMS = frozenset({
    'RS256', 'RS384', 'RS512',
    'PS256', 'PS384', 'PS512',
    'ES256', 'ES384', 'ES512',
    'EdDSA',
})


def allowed_algorithms() -> List[str]:
    algos = [a for a in settings.OIDC_RP_SIGN_ALGOS_ALLOWED if a in ASYMMETRIC_ALGORITHMS]
    return algos or ['RS256']


def unverified_claims(token: Optional[str]) -> Dict[str, Any]:
    """Claims of a token we obtained ourselves from the OP (no signature check)."""
    if not token:
        return {}
    try:
        claims = jwt.decode(token, options={'verify_signature': False, 'verify_exp': False, 'verify_aud': False})
    except PyJWTError:
        return {}
    return claims if isinstance(claims, dict) else {}


def _header(token: str) -> Dict[str, Any]:
    try:
        header = jwt.get_unverified_header(token)
    except PyJWTError:
        raise OIDCTokenError('malformed_token')
    if not isinstance(header, dict):
        raise OIDCTokenError('malformed_token')
    return header


def _key_and_alg(token: str) -> tuple:
    header = _header(token)
    alg = header.get('alg')
    if not alg or alg not in allowed_algorithms():
        raise OIDCTokenError('algorithm_not_allowed')
    return get_provider().key_for(header.get('kid'), alg), alg


def _decode(token: str, *, audience: Optional[Iterable[str]], leeway: int, require: List[str]) -> Dict[str, Any]:
    key, alg = _key_and_alg(token)
    issuer = get_provider().expected_issuer()
    options: Options = {'require': require, 'verify_aud': audience is not None}
    try:
        return jwt.decode(
            token,
            key,
            algorithms=[alg],
            audience=list(audience) if audience is not None else None,
            issuer=issuer,
            leeway=leeway,
            options=options,
        )
    except jwt.ExpiredSignatureError:
        raise OIDCTokenError('token_expired')
    except jwt.ImmatureSignatureError:
        raise OIDCTokenError('token_not_yet_valid')
    except jwt.InvalidAudienceError:
        raise OIDCTokenError('invalid_audience')
    except jwt.InvalidIssuerError:
        raise OIDCTokenError('invalid_issuer')
    except jwt.MissingRequiredClaimError:
        raise OIDCTokenError('missing_claim')
    except PyJWTError:
        raise OIDCTokenError('invalid_signature')


def validate_id_token(token: str, *, nonce: Optional[str] = None) -> Dict[str, Any]:
    """Validate an id token: signature, iss, aud, azp, exp, iat, nonce and ``typ``."""
    client_id = settings.OIDC_RP_CLIENT_ID
    claims = _decode(token, audience=[client_id], leeway=settings.OIDC_ID_TOKEN_LEEWAY_SECONDS,
                     require=['exp', 'iat', 'iss', 'sub', 'aud'])
    aud = claims.get('aud')
    if isinstance(aud, list) and len(aud) > 1 and claims.get('azp') != client_id:
        raise OIDCTokenError('invalid_azp')
    if 'azp' in claims and claims['azp'] != client_id:
        raise OIDCTokenError('invalid_azp')
    if claims.get('typ') not in (None, 'ID'):
        # e.g. an access token (typ "Bearer") presented as an id token
        raise OIDCTokenError('invalid_token_type')
    if nonce is not None and claims.get('nonce') != nonce:
        raise OIDCTokenError('invalid_nonce')
    return claims


def validate_access_token(token: str) -> Dict[str, Any]:
    """Validate a bearer access token: signature, iss, exp/nbf/iat, ``typ`` and audience."""
    claims = _decode(token, audience=None, leeway=settings.OIDC_BEARER_LEEWAY_SECONDS,
                     require=['exp', 'iss', 'sub'])
    expected_type = settings.OIDC_BEARER_TOKEN_TYPE
    if expected_type and claims.get('typ') != expected_type:
        # an id token has aud == client_id too; typ is what tells them apart
        raise OIDCTokenError('invalid_token_type')
    audiences = set(settings.OIDC_BEARER_AUDIENCES or [settings.OIDC_RP_CLIENT_ID])
    aud = claims.get('aud')
    token_aud = set(aud) if isinstance(aud, list) else ({aud} if aud else set())
    if token_aud & audiences:
        return claims
    if settings.OIDC_BEARER_ACCEPT_AZP and claims.get('azp') in audiences:
        return claims
    raise OIDCTokenError('invalid_audience')


def validate_logout_token(token: str) -> Dict[str, Any]:
    """Validate a back-channel logout token (OpenID Connect Back-Channel Logout 1.0, §2.6)."""
    leeway = settings.OIDC_ID_TOKEN_LEEWAY_SECONDS
    claims = _decode(token, audience=[settings.OIDC_RP_CLIENT_ID], leeway=leeway,
                     require=['iat', 'iss', 'aud'])
    events = claims.get('events')
    if not isinstance(events, dict) or constants.BACKCHANNEL_LOGOUT_EVENT not in events:
        raise OIDCTokenError('missing_logout_event')
    if 'nonce' in claims:
        raise OIDCTokenError('nonce_not_allowed')
    if not claims.get('sid') and not claims.get('sub'):
        raise OIDCTokenError('missing_sid_and_sub')
    now = time.time()
    iat = claims.get('iat')
    if not isinstance(iat, (int, float)):
        raise OIDCTokenError('missing_claim')
    if iat > now + leeway or iat < now - settings.OIDC_BACKCHANNEL_LOGOUT_MAX_AGE_SECONDS - leeway:
        raise OIDCTokenError('stale_logout_token')
    return claims


__all__ = [
    'ASYMMETRIC_ALGORITHMS',
    'allowed_algorithms',
    'unverified_claims',
    'validate_access_token',
    'validate_id_token',
    'validate_logout_token',
]
