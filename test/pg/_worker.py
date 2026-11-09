"""Entry point of the separate processes of the Postgres suite (multiprocessing "spawn")."""
import os


def refresh_in_process(db_name, discovery_url, session_id, barrier, results):
    os.environ['DJANGO_SETTINGS_MODULE'] = 'test.pg.settings_pg'
    os.environ['OIDC_PG_DB'] = db_name
    os.environ['NO_PROXY'] = '127.0.0.1,localhost'
    import django
    django.setup()
    from django.db import connections
    from django.test.utils import override_settings

    from oauth2_authcodeflow import sessions

    override_settings(OIDC_OP_DISCOVERY_DOCUMENT_URL=discovery_url).enable()
    try:
        barrier.wait(60)
        results.put(('ok', sessions.access_token_for(session_id)))
    except Exception as e:  # reported to the parent
        results.put(('error', f'{type(e).__name__}: {e}'))
    finally:
        connections.close_all()
