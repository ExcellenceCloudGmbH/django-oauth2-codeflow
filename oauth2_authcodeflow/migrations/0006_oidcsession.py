import uuid

import django.db.models.deletion
from django.conf import settings
from django.db import (
    migrations,
    models,
)


class Migration(migrations.Migration):
    """3.0: tokens move from the Django session into OIDCSession rows. Additive only."""

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('oauth2_authcodeflow', '0005_alter_blacklistedtoken'),
    ]

    operations = [
        migrations.CreateModel(
            name='OIDCSession',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('django_session_key', models.CharField(blank=True, db_index=True, default='', max_length=40)),
                ('issuer', models.CharField(max_length=512)),
                ('client_id', models.CharField(max_length=255)),
                ('sub', models.CharField(max_length=255)),
                ('sid', models.CharField(blank=True, default='', max_length=255)),
                ('claims', models.JSONField(blank=True, default=dict)),
                ('access_token_enc', models.TextField()),
                ('refresh_token_enc', models.TextField(blank=True, null=True)),
                ('id_token_enc', models.TextField(blank=True, null=True)),
                ('access_expires_at', models.DateTimeField()),
                ('refresh_expires_at', models.DateTimeField(blank=True, null=True)),
                ('version', models.PositiveIntegerField(default=1)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('last_refresh_at', models.DateTimeField(blank=True, null=True)),
                ('last_seen_at', models.DateTimeField(blank=True, null=True)),
                ('ended_at', models.DateTimeField(blank=True, db_index=True, null=True)),
                ('end_reason', models.CharField(blank=True, default='', max_length=32)),
                ('migrated_from_legacy', models.BooleanField(default=False)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='oidc_sessions', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'OIDC session',
                'indexes': [
                    models.Index(fields=['issuer', 'sid'], name='oidc_sess_iss_sid_idx'),
                    models.Index(fields=['issuer', 'sub'], name='oidc_sess_iss_sub_idx'),
                    models.Index(fields=['user', 'ended_at'], name='oidc_sess_user_end_idx'),
                    models.Index(fields=['refresh_expires_at'], name='oidc_sess_rexp_idx'),
                ],
            },
        ),
    ]
