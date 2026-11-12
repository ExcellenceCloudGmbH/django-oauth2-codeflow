"""Settings of the Postgres concurrency suite: SQLite ignores ``SELECT ... FOR UPDATE``.

Run: ``pytest test/pg -m postgres --ds=test.pg.settings_pg``
Connection: the libpq variables PGHOST (default /tmp), PGPORT (5432), PGUSER (postgres), PGPASSWORD.
Set OIDC_PG_REQUIRED=1 (CI) to fail instead of skipping when Postgres is unreachable.
"""
import os

from test.settings import *  # noqa: F401,F403

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('OIDC_PG_DB', 'cf3_codeflow'),
        'USER': os.environ.get('PGUSER', 'postgres'),
        'PASSWORD': os.environ.get('PGPASSWORD', ''),
        'HOST': os.environ.get('PGHOST', '/tmp'),
        'PORT': os.environ.get('PGPORT', '5432'),
        'TEST': {'NAME': os.environ.get('OIDC_PG_TEST_DB', 'cf3_test_codeflow')},
    },
}
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
