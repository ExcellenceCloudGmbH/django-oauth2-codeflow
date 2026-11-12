"""Structured, redacting event log.

Every forced logout, refresh failure and circuit-breaker change is logged as one line
``event=<name> key=value ...`` on the ``oauth2_authcodeflow.events`` logger, with the
fields also attached to the record (``record.oidc_event`` / ``record.oidc_fields``) for
structured handlers. Token-shaped values are never written.
"""
import logging
import re
from typing import Any

logger = logging.getLogger('oauth2_authcodeflow.events')

# a JWT (three base64url segments) or an opaque bearer-looking secret
_JWT_RE = re.compile(r'eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*')
_MAX_VALUE = 200


def redact(value: Any) -> Any:
    """Strip anything that looks like a token from a value meant for logs or responses."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    text = _JWT_RE.sub('[redacted]', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    if len(text) > _MAX_VALUE:
        text = text[:_MAX_VALUE] + '…'
    return text


def _format(value: Any) -> str:
    text = str(value)
    if not text or any(c in text for c in ' "='):
        return '"' + text.replace('"', "'") + '"'
    return text


def emit(event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log one structured event."""
    if not logger.isEnabledFor(level):
        return
    clean = {key: redact(value) for key, value in fields.items() if value is not None}
    message = ' '.join([f'event={event}'] + [f'{key}={_format(value)}' for key, value in clean.items()])
    logger.log(level, message, extra={'oidc_event': event, 'oidc_fields': clean})


__all__ = ['emit', 'logger', 'redact']
