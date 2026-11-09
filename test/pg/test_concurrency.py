"""Single-flight refresh on a real database (SQLite ignores ``SELECT ... FOR UPDATE``).

Intent (plan section 6.1, CI): 16 threads on an expired token make exactly one token call;
separate processes make one call; followers inside the skew window never wait; an
``invalid_grant`` under contention ends the sign-in once; a back-channel logout racing a
refresh ends the sign-in; a stale session saved late stays signed in. Also: waits are
bounded, an outage under contention reaches the OP a bounded number of times, and lock
errors of the database (55P03, 40001, 40P01) mean "unavailable", never a logout.
"""
import http.server
import json
import multiprocessing
import threading
import time
from datetime import timedelta
from importlib import import_module
from unittest import mock
from urllib.parse import parse_qs

import pytest
from django.conf import settings as dj_settings
from django.db import (
    OperationalError,
    connection,
    connections,
    transaction,
)
from django.db.models import F
from django.test import (
    Client,
    override_settings,
)
from django.utils import timezone

from oauth2_authcodeflow import (
    sessions,
    signals,
)
from oauth2_authcodeflow.exceptions import (
    OIDCAuthUnavailable,
    OIDCSessionEnded,
)
from oauth2_authcodeflow.models import OIDCSession
from oauth2_authcodeflow.testing import create_oidc_session

pytestmark = [pytest.mark.postgres, pytest.mark.django_db(transaction=True), pytest.mark.timeout(120)]

XHR = {'HTTP_SEC_FETCH_MODE': 'cors'}


# --- helpers --------------------------------------------------------------------------
def _expire(row, seconds=-10):
    OIDCSession.objects.filter(pk=row.pk).update(access_expires_at=timezone.now() + timedelta(seconds=seconds))
    return OIDCSession.objects.get(pk=row.pk)


def _slow_tokens(op, sub, delay):
    def answer(data):
        time.sleep(delay)
        return op.issue_tokens(sub)
    return answer


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, 'condition not reached'
        time.sleep(0.01)


class _Background(threading.Thread):
    def __init__(self, func):
        super().__init__(daemon=True)
        self.func, self.result, self.error = func, None, None

    def run(self):
        try:
            self.result = self.func()
        except Exception as e:
            self.error = e
        finally:
            connections.close_all()


def _in_threads(count, pk, call=sessions.ensure_fresh):
    """``call(row)`` in ``count`` threads (own connection, own row instance) released
    together. Returns [(status, value or exception, seconds)]."""
    barrier = threading.Barrier(count)
    results = [None] * count

    def worker(index):
        try:
            row = OIDCSession.objects.get(pk=pk)
            barrier.wait(10)
            started = time.monotonic()
            try:
                results[index] = ('ok', call(row), time.monotonic() - started)
            except Exception as e:
                results[index] = ('error', e, time.monotonic() - started)
        finally:
            connections.close_all()

    threads = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert all(result is not None for result in results), results
    return results


def _sqlstate(error):
    cause = error.__cause__
    return getattr(cause, 'pgcode', None) or getattr(cause, 'sqlstate', None)


class _StubOP:
    """The fake OP over real HTTP, for other processes: discovery, JWKS and a slow token endpoint."""

    def __init__(self, op, delay):
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _reply(self, status, body):
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path.endswith('/.well-known/openid-configuration'):
                    self._reply(200, dict(op.discovery(), token_endpoint=f'{stub.base}/token', jwks_uri=f'{stub.base}/certs'))
                elif self.path == '/certs':
                    self._reply(200, op.jwks())
                else:
                    self._reply(404, {})

            def do_POST(self):
                length = int(self.headers.get('Content-Length') or 0)
                data = {key: values[0] for key, values in parse_qs(self.rfile.read(length).decode()).items()}
                time.sleep(delay)
                response = op._token(data)
                self._reply(response.status_code, response.json())

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        self.discovery_url = f'{self.base}/realms/test/.well-known/openid-configuration'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


# --- P1 to P10 ------------------------------------------------------------------------
def test_p1_sixteen_threads_on_an_expired_token_make_one_token_call(user, oidc_op):
    row = _expire(create_oidc_session(Client(), user))
    oidc_op.script_token(_slow_tokens(oidc_op, row.sub, 0.3))
    results = _in_threads(16, row.pk)
    assert [status for status, _, _ in results] == ['ok'] * 16, results
    assert oidc_op.refresh_calls() == 1
    tokens = {value.access_token for _, value, _ in results}
    assert len(tokens) == 1 and tokens != {row.access_token}
    assert OIDCSession.objects.get(pk=row.pk).version == row.version + 1


def test_p2_separate_processes_make_one_token_call(user, oidc_op):
    from test.pg._worker import refresh_in_process
    row = _expire(create_oidc_session(Client(), user))
    stub = _StubOP(oidc_op, delay=0.5)
    context = multiprocessing.get_context('spawn')
    workers = 4
    barrier, results = context.Barrier(workers + 1), context.Queue()
    processes = [
        context.Process(target=refresh_in_process, args=(connection.settings_dict['NAME'], stub.discovery_url, str(row.pk), barrier, results))
        for _ in range(workers)
    ]
    try:
        for process in processes:
            process.start()
        barrier.wait(90)  # every process is set up: release them together
        answers = [results.get(timeout=60) for _ in processes]
    finally:
        for process in processes:
            process.join(30)
        stub.stop()
    assert [status for status, _ in answers] == ['ok'] * workers, answers
    assert len({token for _, token in answers}) == 1
    assert oidc_op.refresh_calls() == 1
    assert OIDCSession.objects.get(pk=row.pk).version == row.version + 1


def test_p3_followers_inside_the_skew_window_never_wait(user, oidc_op):
    row = create_oidc_session(Client(), user, access_expires_in=30)  # usable, inside the 60 s skew
    oidc_op.script_token(_slow_tokens(oidc_op, row.sub, 1.5))
    leader = _Background(lambda: sessions.ensure_fresh(OIDCSession.objects.get(pk=row.pk)))
    leader.start()
    _wait_for(lambda: oidc_op.refresh_calls() == 1)
    results = _in_threads(8, row.pk)
    leader.join(10)
    assert leader.error is None and leader.result.version == row.version + 1
    for status, value, seconds in results:
        assert status == 'ok' and value.access_token == row.access_token and seconds < 0.5, (status, value, seconds)
    assert oidc_op.refresh_calls() == 1


def test_p4_waiting_for_a_slow_refresh_is_bounded_and_keeps_the_sign_in(user, oidc_op):
    row = _expire(create_oidc_session(Client(), user))
    oidc_op.script_token(_slow_tokens(oidc_op, row.sub, 3))
    with override_settings(OIDC_REFRESH_WAIT_SECONDS=1):
        leader = _Background(lambda: sessions.ensure_fresh(OIDCSession.objects.get(pk=row.pk)))
        leader.start()
        _wait_for(lambda: oidc_op.refresh_calls() == 1)
        results = _in_threads(4, row.pk)
        leader.join(10)
    for status, value, seconds in results:
        assert status == 'error' and isinstance(value, OIDCAuthUnavailable), value
        assert 0.8 <= seconds < 2.5, seconds
    assert leader.error is None
    stored = OIDCSession.objects.get(pk=row.pk)
    assert stored.ended_at is None and stored.version == row.version + 1
    assert oidc_op.refresh_calls() == 1


def test_p5_invalid_grant_under_contention_ends_the_sign_in_once(user, oidc_op):
    row = _expire(create_oidc_session(Client(), user))

    def invalid_grant(data):
        time.sleep(0.3)
        return (400, {'error': 'invalid_grant', 'error_description': 'Session not active'})
    oidc_op.script_token(invalid_grant)
    ended = []
    signals.oidc_session_ended.connect(lambda **kw: ended.append(kw['reason']), weak=False, dispatch_uid='p5')
    try:
        results = _in_threads(8, row.pk)
    finally:
        signals.oidc_session_ended.disconnect(dispatch_uid='p5')
    assert all(status == 'error' and isinstance(value, OIDCSessionEnded) for status, value, _ in results), results
    assert oidc_op.refresh_calls() == 1
    assert ended == ['refresh_invalid_grant']
    assert OIDCSession.objects.get(pk=row.pk).end_reason == 'refresh_invalid_grant'


def test_p6_backchannel_logout_racing_a_refresh_ends_the_sign_in(user, oidc_op):
    client = Client()
    row = _expire(create_oidc_session(client, user))
    oidc_op.script_token(_slow_tokens(oidc_op, row.sub, 1))
    leader = _Background(lambda: sessions.ensure_fresh(OIDCSession.objects.get(pk=row.pk)))
    leader.start()
    _wait_for(lambda: oidc_op.refresh_calls() == 1)  # the refresh holds the row lock now
    response = Client().post('/oidc/backchannel-logout', {'logout_token': oidc_op.logout_token(sid=row.sid)})
    leader.join(10)
    assert response.status_code == 200
    stored = OIDCSession.objects.get(pk=row.pk)
    assert stored.ended_at is not None and stored.end_reason == 'backchannel_logout'
    assert client.get('/api/data/', **XHR).status_code == 401


def test_p7_a_stale_session_saved_late_stays_signed_in(user, oidc_op):
    client = Client()
    row = create_oidc_session(client, user)
    stale = import_module(dj_settings.SESSION_ENGINE).SessionStore(session_key=client.session.session_key)
    assert stale.get('_oidc_session_id') == str(row.pk)  # snapshot of a slow request
    latest = {}

    def rotating(data):  # the OP rotates refresh tokens and refuses a used one
        if latest and data.get('refresh_token') != latest['refresh_token']:
            return (400, {'error': 'invalid_grant', 'error_description': 'Stale token'})
        tokens = oidc_op.issue_tokens(row.sub)
        latest['refresh_token'] = tokens['refresh_token']
        return tokens
    oidc_op.script_token(rotating, rotating)
    _expire(row)
    assert client.get('/api/data/', **XHR).status_code == 200  # refresh 1 rotates the refresh token
    stale['late_write'] = True
    stale.save()  # the slow request saves its old snapshot of the session
    _expire(row)
    assert client.get('/api/data/', **XHR).status_code == 200  # refresh 2 uses the rotated token
    assert oidc_op.refresh_calls() == 2
    assert OIDCSession.objects.get(pk=row.pk).ended_at is None


def test_p8_an_outage_under_contention_reaches_the_op_a_bounded_number_of_times(user, oidc_op):
    row = _expire(create_oidc_session(Client(), user))
    oidc_op.script_token(*[(503, 'Service Unavailable')] * 16)
    results = _in_threads(16, row.pk)
    assert all(status == 'error' and isinstance(value, OIDCAuthUnavailable) for status, value, _ in results), results
    assert 1 <= oidc_op.refresh_calls() <= 5  # the circuit breaker opens after 5 failures
    assert OIDCSession.objects.get(pk=row.pk).ended_at is None


def test_p11_a_refresh_in_progress_does_not_block_rows_that_reference_the_sign_in(user, oidc_op):
    """A row with a foreign key to the sign-in (a dashboard binding, say) is committed while a
    refresh holds the sign-in's row across the provider call. Postgres checks that key with
    FOR KEY SHARE, which a FOR UPDATE lock would block until the refresh commits; the refresh
    takes FOR NO KEY UPDATE, so the check passes at once."""
    row = _expire(create_oidc_session(Client(), user))
    oidc_op.script_token(_slow_tokens(oidc_op, row.sub, 1.5))
    leader = _Background(lambda: sessions.ensure_fresh(OIDCSession.objects.get(pk=row.pk)))
    leader.start()
    _wait_for(lambda: oidc_op.refresh_calls() == 1)  # the leader holds the row now

    def foreign_key_check():
        table = connection.ops.quote_name(OIDCSession._meta.db_table)
        column = connection.ops.quote_name(OIDCSession._meta.pk.column)
        with transaction.atomic():
            with connection.cursor() as cursor:
                cursor.execute(f'SELECT 1 FROM {table} WHERE {column} = %s FOR KEY SHARE NOWAIT', [row.pk])
                return cursor.fetchone()

    checker = _Background(foreign_key_check)
    checker.start()
    checker.join(10)
    leader.join(10)
    assert checker.error is None, checker.error  # under FOR UPDATE: 55P03, lock not available
    assert checker.result == (1,)
    assert leader.error is None and leader.result.version == row.version + 1


def test_p9_a_row_locked_by_another_process(user, oidc_op):
    usable = create_oidc_session(Client(), user, access_expires_in=30)
    expired = _expire(create_oidc_session(Client(), user))
    locked, release = threading.Event(), threading.Event()

    def hold():
        with transaction.atomic():
            list(OIDCSession.objects.select_for_update().filter(pk__in=[usable.pk, expired.pk]).values_list('pk', flat=True))
            locked.set()
            release.wait(20)
    holder = _Background(hold)
    holder.start()
    assert locked.wait(5)
    try:
        started = time.monotonic()
        served = sessions.ensure_fresh(OIDCSession.objects.get(pk=usable.pk))  # FOR UPDATE NOWAIT: 55P03
        assert time.monotonic() - started < 0.5
        assert served.access_token == usable.access_token
        with override_settings(OIDC_REFRESH_WAIT_SECONDS=1):
            started = time.monotonic()
            with pytest.raises(OIDCAuthUnavailable):
                sessions.ensure_fresh(OIDCSession.objects.get(pk=expired.pk))  # lock_timeout: 55P03
            assert 0.8 <= time.monotonic() - started < 3
        assert oidc_op.refresh_calls() == 0
    finally:
        release.set()
        holder.join(10)
    fresh = sessions.ensure_fresh(OIDCSession.objects.get(pk=expired.pk))
    assert fresh.version == expired.version + 1 and oidc_op.refresh_calls() == 1
    assert OIDCSession.objects.filter(ended_at__isnull=False).count() == 0


def _deadlock(pk_a, pk_b):
    barrier, errors = threading.Barrier(2), []

    def lock(first, second):
        def run():
            try:
                with transaction.atomic():
                    list(OIDCSession.objects.select_for_update().filter(pk=first).values_list('pk', flat=True))
                    barrier.wait(5)
                    list(OIDCSession.objects.select_for_update().filter(pk=second).values_list('pk', flat=True))
            except OperationalError as e:
                errors.append(e)
        return _Background(run)
    threads = [lock(pk_a, pk_b), lock(pk_b, pk_a)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)
    return errors


def _serialization_failure(pk):
    snapshot, updated, errors = threading.Event(), threading.Event(), []

    def run():
        try:
            with transaction.atomic():
                with connection.cursor() as cursor:
                    cursor.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
                list(OIDCSession.objects.filter(pk=pk).values_list('version', flat=True))
                snapshot.set()
                updated.wait(5)
                OIDCSession.objects.filter(pk=pk).update(version=F('version') + 1)
        except OperationalError as e:
            errors.append(e)
    reader = _Background(run)
    reader.start()
    assert snapshot.wait(5)
    OIDCSession.objects.filter(pk=pk).update(last_seen_at=timezone.now())  # a concurrent committed write
    updated.set()
    reader.join(10)
    return errors


def test_p10_lock_errors_of_the_database_mean_unavailable_never_a_logout(user, oidc_op):
    a = create_oidc_session(Client(), user)
    b = create_oidc_session(Client(), user)
    deadlocks = _deadlock(a.pk, b.pk)
    failures = _serialization_failure(a.pk)
    assert len(deadlocks) == 1 and len(failures) == 1, (deadlocks, failures)
    assert {_sqlstate(deadlocks[0]), _sqlstate(failures[0])} == {'40P01', '40001'}
    assert sessions._is_contention(deadlocks[0]) and sessions._is_contention(failures[0])
    row = _expire(a)
    with mock.patch.object(OIDCSession, 'save', side_effect=failures[0]):  # the database aborts the refresh
        with pytest.raises(OIDCAuthUnavailable):
            sessions.ensure_fresh(row)
    stored = OIDCSession.objects.get(pk=a.pk)
    assert stored.ended_at is None and stored.version == row.version and stored.access_token == row.access_token
