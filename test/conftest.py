import pytest

pytest_plugins = ['oauth2_authcodeflow.testing.pytest_plugin']


@pytest.fixture(autouse=True)
def _isolated_state():
    from django.core.cache import cache

    from oauth2_authcodeflow import sessions
    from oauth2_authcodeflow.breaker import reset_breakers
    cache.clear()
    reset_breakers()
    sessions.reset_refresh_failures()
    yield
    reset_breakers()
    sessions.reset_refresh_failures()


@pytest.fixture
def user(db):
    from django.contrib.auth import get_user_model
    return get_user_model().objects.create_user(username='alice', email='alice@example.test', password='pw-alice-123')
