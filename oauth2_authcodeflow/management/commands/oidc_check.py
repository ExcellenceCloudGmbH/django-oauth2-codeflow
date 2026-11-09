import json

from django.core.management.base import (
    BaseCommand,
    CommandError,
)
from requests import RequestException

from oauth2_authcodeflow.conf import settings
from oauth2_authcodeflow.crypto import get_cipher
from oauth2_authcodeflow.exceptions import (
    OIDCAuthUnavailable,
    OIDCConfigurationError,
)
from oauth2_authcodeflow.http import get_client
from oauth2_authcodeflow.provider import get_provider


class Command(BaseCommand):
    help = (
        "Check the OIDC configuration against the OP: effective endpoints, issuer equality, "
        "signing keys and token-endpoint reachability. Exits non-zero on a problem."
    )

    def add_arguments(self, parser):
        parser.add_argument('--json', action='store_true', help="machine-readable output")

    def handle(self, *args, **options):
        report = {'discovery_url': settings.OIDC_OP_DISCOVERY_DOCUMENT_URL, 'configured_issuer': settings.OIDC_OP_ISSUER}
        problems = []
        provider = get_provider()
        try:
            md = provider.metadata()
        except (OIDCAuthUnavailable, OIDCConfigurationError) as e:
            problems.append(f"discovery: {e}")
            md = None
        if md is not None:
            report.update({
                'issuer': md.issuer,
                'discovered_issuer': md.discovered_issuer,
                'authorization_endpoint': md.authorization_endpoint,
                'token_endpoint': md.token_endpoint,
                'userinfo_endpoint': md.userinfo_endpoint,
                'jwks_uri': md.jwks_uri,
                'end_session_endpoint': md.end_session_endpoint,
                'revocation_endpoint': md.revocation_endpoint,
            })
            if settings.OIDC_OP_ISSUER and md.discovered_issuer and settings.OIDC_OP_ISSUER.rstrip('/') != md.discovered_issuer.rstrip('/'):
                problems.append("issuer: OIDC_OP_ISSUER differs from the discovery issuer")
            try:
                keys = provider._jwks(force=True)
                report['signing_keys'] = len(keys)
                if not keys:
                    problems.append("jwks: no signing key")
            except (OIDCAuthUnavailable, OIDCConfigurationError) as e:
                problems.append(f"jwks: {e}")
            try:
                resp = get_client().post(md.token_endpoint, data={'grant_type': 'refresh_token', 'refresh_token': 'oidc_check'},
                                         headers={'Accept': 'application/json'})
                try:
                    body = resp.json()
                except ValueError:
                    body = None
                reachable = isinstance(body, dict) and 'error' in body
                report['token_endpoint_status'] = resp.status_code
                report['token_endpoint_reachable'] = reachable
                if not reachable:
                    problems.append(f"token endpoint: unexpected answer (HTTP {resp.status_code})")
            except RequestException as e:
                problems.append(f"token endpoint: {type(e).__name__}")
        cipher = get_cipher()
        report['token_encryption'] = 'derived from SECRET_KEY' if cipher.derived else 'explicit keys'
        report['problems'] = problems
        if options['json']:
            self.stdout.write(json.dumps(report, indent=2, default=str))
        else:
            for key, value in report.items():
                if key != 'problems':
                    self.stdout.write(f"{key}: {value}")
            for problem in problems:
                self.stderr.write(f"PROBLEM {problem}")
        if problems:
            raise CommandError(f"{len(problems)} problem(s) found")
        self.stdout.write(self.style.SUCCESS("OIDC configuration OK"))
