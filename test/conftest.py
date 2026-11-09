import pytest
from django.contrib.sessions.middleware import SessionMiddleware


@pytest.fixture
def sf():
    """
    Session factory
    takes a request as argument and create a session (not saved) on it
    """
    def session_factory(request):
        middleware = SessionMiddleware(lambda x: None)
        middleware.process_request(request)
        return request.session
    yield session_factory
