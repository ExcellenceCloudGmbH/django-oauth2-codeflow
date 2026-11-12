from django.core.exceptions import DisallowedHost
from django.core.management.base import (
    BaseCommand,
    CommandError,
)
from django.http import HttpRequest
from django.urls import reverse
from django.urls.exceptions import NoReverseMatch

from oauth2_authcodeflow.conf import constants


class Command(BaseCommand):
    help = "Show the URLs to configure on the OP client: redirect URI, post-logout redirect URI, back-channel logout URL"

    def add_arguments(self, parser):
        parser.add_argument('host', type=str, help="public host name (with the port if it is not the default one)")
        parser.add_argument('--secure', action='store_true', help="specify that HTTPS will be used as public access")

    def handle(self, *args, **options):
        host = options['host']
        scheme = 'https' if options['secure'] else 'http'
        try:
            callback = f"{scheme}://{host}{reverse(constants.OIDC_URL_CALLBACK_NAME)}"
            backchannel = f"{scheme}://{host}{reverse(constants.OIDC_URL_BACKCHANNEL_LOGOUT_NAME)}"
        except NoReverseMatch as e:
            raise CommandError(str(e))
        request = HttpRequest()
        request.META['HTTP_HOST'] = host
        try:
            request.get_host()
        except DisallowedHost:
            self.stderr.write(f"warning: {host!r} is not in ALLOWED_HOSTS; Django will reject the OP's redirects to it")
        self.stdout.write(f"redirect_url: {callback}")
        self.stdout.write(f"post_logout_redirect_url: {callback}")
        self.stdout.write(f"backchannel_logout_url: {backchannel}")
