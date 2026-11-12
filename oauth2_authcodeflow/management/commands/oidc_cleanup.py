from datetime import timedelta

from django.conf import settings as dj_settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from oauth2_authcodeflow import sessions
from oauth2_authcodeflow.models import (
    BlacklistedToken,
    OIDCSession,
)


class Command(BaseCommand):
    help = (
        "End sign-ins whose refresh token expired or whose Django session is gone, delete "
        "ended sign-ins older than --ended-older-than days and purge expired blacklist rows. "
        "Run it daily together with 'clearsessions'."
    )

    def add_arguments(self, parser):
        parser.add_argument('--ended-older-than', type=int, default=7, help="days to keep ended sign-ins (default 7)")
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        now = timezone.now()
        dry = options['dry_run']
        active = OIDCSession.objects.filter(ended_at__isnull=True)

        expired = list(active.filter(refresh_expires_at__lt=now))
        no_refresh = list(active.filter(refresh_token_enc__isnull=True, access_expires_at__lt=now))
        orphaned = []
        if dj_settings.SESSION_ENGINE in ('django.contrib.sessions.backends.db', 'django.contrib.sessions.backends.cached_db'):
            from django.contrib.sessions.models import Session
            for row in active.exclude(django_session_key='').only('pk', 'django_session_key').iterator():
                if not Session.objects.filter(session_key=row.django_session_key, expire_date__gt=now).exists():
                    orphaned.append(row)
        old = OIDCSession.objects.filter(ended_at__lt=now - timedelta(days=options['ended_older_than']))

        ended = 0
        if not dry:
            for row in expired:
                ended += sessions.end_session(row, 'refresh_expired')
            for row in no_refresh:
                ended += sessions.end_session(row, 'access_expired')
            for row in orphaned:
                ended += sessions.end_session(row, 'orphaned', delete_django_session=False)
            deleted, _ = old.delete()
            purged = BlacklistedToken.purge()
        else:
            deleted, purged = old.count(), BlacklistedToken.objects.filter(expires_at__lte=now).count()
        prefix = '[dry-run] ' if dry else ''
        self.stdout.write(
            f"{prefix}ended {ended if not dry else len(expired) + len(no_refresh) + len(orphaned)} sign-in(s) "
            f"(refresh expired {len(expired)}, access expired {len(no_refresh)}, orphaned {len(orphaned)}); "
            f"deleted {deleted} ended sign-in(s); purged {purged} blacklisted token(s)"
        )
