"""Test helpers for projects that use oauth2_authcodeflow.

- ``create_oidc_session(client_or_request, user, ...)``: a signed-in OIDC session without an
  OP (replaces hand-crafted ``session["oidc_*"]`` keys, which 3.0 no longer reads).
- ``FakeOP``: an in-process OpenID provider (discovery, JWKS, token, userinfo, revocation)
  with scriptable token-endpoint answers.
- ``make_signed_token(claims)``: an RS256 token signed with the test key FakeOP publishes.
- pytest fixture ``oidc_op`` (plugin loaded automatically through the ``pytest11`` entry point).

The names are imported lazily: pytest loads the plugin before Django is configured, and
importing this package must not touch models or settings.
"""
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:  # pragma: no cover
    from .fake_op import (
        FakeOP,
        FakeResponse,
        make_signed_token,
        test_private_key,
    )
    from .helpers import create_oidc_session

__all__ = ['FakeOP', 'FakeResponse', 'create_oidc_session', 'make_signed_token', 'test_private_key']


def __getattr__(name: str) -> Any:
    if name == 'create_oidc_session':
        from .helpers import create_oidc_session
        return create_oidc_session
    if name in ('FakeOP', 'FakeResponse', 'make_signed_token', 'test_private_key'):
        from . import fake_op
        return getattr(fake_op, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
