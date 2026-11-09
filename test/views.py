from oauth2_authcodeflow.utils import login_required
from django.http import (
    HttpResponse,
    JsonResponse,
)
from rest_framework import exceptions
from rest_framework.response import Response
from rest_framework.views import APIView

from oauth2_authcodeflow.exceptions import (
    OIDCAuthUnavailable,
    OIDCSessionEnded,
)
from oauth2_authcodeflow.tokens import get_access_token


def protected(request):
    return HttpResponse(f"ok {request.user.get_username()}")


def public(request):
    return HttpResponse("public")


def health(request):
    return HttpResponse("healthy")


def api_data(request):
    ctx = getattr(request, 'oidc', None)
    return JsonResponse({
        'user': request.user.get_username(),
        'has_oidc': ctx is not None,
        'bearer': bool(ctx and ctx.is_bearer),
        'roles': ctx.client_roles() if ctx else [],
    })


def api_fresh(request):
    token = get_access_token(request, min_ttl=int(request.GET.get('min_ttl', '0')))
    return JsonResponse({'has_token': bool(token), 'version': request.oidc.version if request.oidc else None})


def api_raise(request):
    kind = request.GET.get('kind')
    if kind == 'ended':
        raise OIDCSessionEnded('test')
    raise OIDCAuthUnavailable('test', retry_after=7)


class DRFData(APIView):
    def get(self, request):
        return Response({'user': request.user.get_username()})

    def post(self, request):
        return Response({'posted': True})


class DRFDenied(APIView):
    def get(self, request):
        raise exceptions.PermissionDenied()


class DRFRaise(APIView):
    def get(self, request):
        if request.query_params.get('kind') == 'ended':
            raise OIDCSessionEnded('test')
        raise OIDCAuthUnavailable('test', retry_after=3)


@login_required
def decorated(request):
    return HttpResponse('decorated')
