"""HTTP client for every call to the OpenID provider.

One ``requests.Session`` per process with a connection pool. Headers are passed per call
and the shared session is never mutated, so concurrent requests of different users can
never send each other's tokens. Cookies are refused, timeouts are always applied and
redirects are not followed.
"""
import threading
from http.cookiejar import DefaultCookiePolicy
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Union,
)

import requests
from django.core.signals import setting_changed
from requests.adapters import HTTPAdapter

from .conf import settings

Timeout = Union[float, Tuple[float, float]]


def http_timeout() -> Timeout:
    value = settings.OIDC_HTTP_TIMEOUT
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("OIDC_HTTP_TIMEOUT must be a number or a (connect, read) pair")
        return float(value[0]), float(value[1])
    return float(value)


class OPClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=settings.OIDC_HTTP_POOL_MAXSIZE)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        # never store or send cookies on OP calls
        self.session.cookies.set_policy(DefaultCookiePolicy(allowed_domains=[]))
        self.session.trust_env = True

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        auth: Optional[Tuple[str, str]] = None,
    ) -> requests.Response:
        return self.session.request(
            method,
            url,
            headers=dict(headers or {}),
            data=data,
            params=params,
            auth=auth,
            timeout=http_timeout(),
            verify=settings.OIDC_CA_BUNDLE,
            proxies=settings.OIDC_PROXY or None,
            allow_redirects=False,
        )

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request('GET', url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request('POST', url, **kwargs)


_lock = threading.Lock()
_client: Optional[OPClient] = None


def get_client() -> OPClient:
    global _client
    client = _client
    if client is None:
        with _lock:
            if _client is None:
                _client = OPClient()
            client = _client
    return client


def set_client(client: Optional[OPClient]) -> None:
    """Replace the process client (tests, ``testing.FakeOP``)."""
    global _client
    with _lock:
        _client = client


def _reset(**kwargs: Any) -> None:
    # only a real client is rebuilt; a test double installed with set_client() stays
    if str(kwargs.get('setting', '')).startswith(('OIDC_HTTP', 'OIDC_PROXY', 'OIDC_CA_BUNDLE')) and type(_client) is OPClient:
        set_client(None)


setting_changed.connect(_reset)

__all__ = ['OPClient', 'get_client', 'http_timeout', 'set_client']
