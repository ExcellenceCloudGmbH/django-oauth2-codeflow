"""pytest plugin (entry point ``pytest11``): the ``oidc_op`` fixture.

Imports nothing from Django at module import time, so it is harmless in projects that
do not use it.
"""
from typing import Iterator

import pytest


@pytest.fixture
def oidc_op() -> Iterator['object']:
    """A FakeOP installed for one test, with the discovery URL and issuer settings."""
    from django.test import override_settings

    from .fake_op import FakeOP
    op = FakeOP()
    with override_settings(**op.django_settings()):
        op.install()
        try:
            yield op
        finally:
            op.uninstall()
