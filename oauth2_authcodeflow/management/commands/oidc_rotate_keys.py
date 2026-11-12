from django.core.management.base import BaseCommand

from oauth2_authcodeflow import sessions
from oauth2_authcodeflow.crypto import (
    TokenDecryptError,
    get_cipher,
)
from oauth2_authcodeflow.models import OIDCSession

FIELDS = ('access_token_enc', 'refresh_token_enc', 'id_token_enc')


class Command(BaseCommand):
    help = (
        "Re-encrypt every stored token with the current primary key (the first of "
        "OIDC_TOKEN_ENCRYPTION_KEYS, or the key derived from SECRET_KEY). Run it after adding a "
        "new key and before removing the old one."
    )

    def add_arguments(self, parser):
        parser.add_argument('--batch-size', type=int, default=500)
        parser.add_argument('--end-undecryptable', action='store_true',
                            help="end active sign-ins whose tokens no key can decrypt")

    def handle(self, *args, **options):
        cipher = get_cipher()
        rotated = failed = 0
        for row in OIDCSession.objects.only('pk', 'ended_at', *FIELDS).iterator(chunk_size=options['batch_size']):
            updates = {}
            try:
                for field in FIELDS:
                    value = getattr(row, field)
                    if value:
                        updates[field] = cipher.rotate(value)
            except TokenDecryptError:
                failed += 1
                if options['end_undecryptable'] and row.ended_at is None:
                    sessions.end_session(row, 'decrypt_failed')
                continue
            if updates:
                OIDCSession.objects.filter(pk=row.pk).update(**updates)
                rotated += 1
        self.stdout.write(f"re-encrypted {rotated} sign-in(s); {failed} could not be decrypted")
