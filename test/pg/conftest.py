import os

import pytest


def _unavailable_reason():
    from django.conf import settings
    db = settings.DATABASES['default']
    if db['ENGINE'] != 'django.db.backends.postgresql':
        return 'the Postgres suite runs with --ds=test.pg.settings_pg'
    try:
        import psycopg2
    except ImportError:  # pragma: no cover
        return 'psycopg2 is not installed'
    try:
        psycopg2.connect(dbname='postgres', user=db['USER'], password=db['PASSWORD'] or None,
                         host=db['HOST'], port=db['PORT'], connect_timeout=3).close()
    except psycopg2.OperationalError as e:
        return f'Postgres is unreachable: {str(e).strip()}'
    return None


def pytest_collection_modifyitems(config, items):
    pg_items = [item for item in items if item.get_closest_marker('postgres')]
    if not pg_items:
        return
    reason = _unavailable_reason()
    if reason is None:
        return
    if os.environ.get('OIDC_PG_REQUIRED'):
        raise pytest.UsageError(reason)
    for item in pg_items:
        item.add_marker(pytest.mark.skip(reason=reason))
