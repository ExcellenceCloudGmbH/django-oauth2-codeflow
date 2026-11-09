from typing import (
    Callable,
    TypeVar,
)
from urllib.parse import urlencode

from django.contrib.auth.decorators import login_required as orig_login_required
from django.http.response import HttpResponseBase
from django.urls import reverse_lazy
from django.utils.text import format_lazy

from .conf import (
    constants,
    settings,
)

_VIEW = TypeVar("_VIEW", bound=Callable[..., HttpResponseBase])


def login_required(function: _VIEW, fail: str = '/') -> _VIEW:
    """``django.contrib.auth.decorators.login_required`` pointing to the OIDC login view."""
    return orig_login_required(
        function,
        redirect_field_name=settings.OIDC_REDIRECT_OK_FIELD_NAME,
        login_url=format_lazy(
            '{url}?' + urlencode({settings.OIDC_REDIRECT_ERROR_FIELD_NAME: fail}),
            url=reverse_lazy(constants.OIDC_URL_AUTHENTICATION_NAME),
        ),
    )


__all__ = ['login_required']
