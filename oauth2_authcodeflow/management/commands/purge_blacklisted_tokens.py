import warnings

from django.core.management.base import BaseCommand

from oauth2_authcodeflow.models import BlacklistedToken


class Command(BaseCommand):
    help = "Deprecated (3.0 writes no blacklist): purge the blacklisted tokens with passed expiry dates. Use oidc_cleanup."

    def handle(self, *args, **options):
        warnings.warn("purge_blacklisted_tokens is deprecated, use oidc_cleanup", DeprecationWarning, stacklevel=2)
        nb = BlacklistedToken.purge()
        self.stdout.write(f"{nb} blacklisted tokens purged")
