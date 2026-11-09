"""The sign-in service: refresh, error taxonomy, ending, migration helpers."""
import time
from datetime import timedelta

import pytest
from django.contrib.sessions.models import Session
from django.db import transaction
from django.test import override_settings
from django.utils import timezone

from oauth2_authcodeflow import (
    sessions,
    signals,
)
from oauth2_authcodeflow.breaker import get_breaker
from oauth2_authcodeflow.exceptions import (
    OIDCAuthUnavailable,
    OIDCSessionEnded,
)
from oauth2_authcodeflow.locks import keyed_lock
from oauth2_authcodeflow.models import OIDCSession
from oauth2_authcodeflow.testing import create_oidc_session

pytestmark = pytest.mark.django_db

INVALID_GRANT = (400, {'error': 'invalid_grant', 'error_description': 'Session not active'})


@pytest.fixture
def row(client, user, oidc_op):
    return create_oidc_session(client, user, access_expires_in=30)  # inside the 60 s skew window


def _expire(row, seconds=-10):
    OIDCSession.objects.filter(pk=row.pk).update(access_expires_at=timezone.now() + timedelta(seconds=seconds))
    return OIDCSession.objects.get(pk=row.pk)


def test_tokens_are_encrypted_at_rest(row):
    raw = OIDCSession.objects.filter(pk=row.pk).values_list('access_token_enc', 'refresh_token_enc', 'id_token_enc').get()
    assert all(value and not value.startswith('eyJ') for value in raw)
    assert row.access_token.startswith('eyJ')


def test_fast_path_makes_no_call(client, user, oidc_op):
    fresh = create_oidc_session(client, user, access_expires_in=600)
    assert sessions.ensure_fresh(fresh) is fresh
    assert oidc_op.refresh_calls() == 0


def test_refresh_success_updates_tokens_and_version(row, oidc_op):
    received = []
    signals.oidc_session_refreshed.connect(lambda **kw: received.append(kw['session'].pk), weak=False, dispatch_uid='t1')
    try:
        old_access = row.access_token
        fresh = sessions.ensure_fresh(row)
    finally:
        signals.oidc_session_refreshed.disconnect(dispatch_uid='t1')
    assert oidc_op.refresh_calls() == 1
    assert fresh.version == row.version + 1 and fresh.access_token != old_access
    assert fresh.remaining() > 200 and fresh.last_refresh_at is not None


def test_invalid_grant_ends_the_sign_in(row, oidc_op, client, django_capture_on_commit_callbacks):
    ended = []
    signals.oidc_session_ended.connect(lambda **kw: ended.append(kw['reason']), weak=False, dispatch_uid='t2')
    key = client.session.session_key
    oidc_op.script_token(INVALID_GRANT)
    try:
        with django_capture_on_commit_callbacks(execute=True):
            with pytest.raises(OIDCSessionEnded, match='refresh_invalid_grant'):
                sessions.ensure_fresh(_expire(row))
    finally:
        signals.oidc_session_ended.disconnect(dispatch_uid='t2')
    stored = OIDCSession.objects.get(pk=row.pk)
    assert stored.ended_at and stored.end_reason == 'refresh_invalid_grant'
    assert ended == ['refresh_invalid_grant']
    assert not Session.objects.filter(session_key=key).exists()


@pytest.mark.parametrize('answer', [(503, 'Service Unavailable'), (429, '<html>rate limited</html>'), (403, '<html>WAF</html>')])
def test_transient_failure_keeps_the_sign_in_and_serves_a_valid_token(row, oidc_op, answer):
    oidc_op.script_token(answer)
    assert sessions.ensure_fresh(row).pk == row.pk  # still valid for 30 s: served
    assert OIDCSession.objects.get(pk=row.pk).ended_at is None


def test_transient_failure_with_an_expired_token_is_unavailable_not_logout(row, oidc_op):
    oidc_op.script_token((502, 'Bad Gateway'))
    with pytest.raises(OIDCAuthUnavailable):
        sessions.ensure_fresh(_expire(row))
    assert OIDCSession.objects.get(pk=row.pk).ended_at is None


def test_configuration_error_keeps_the_sign_in_and_opens_the_breaker(row, oidc_op):
    oidc_op.script_token((401, {'error': 'invalid_client'}))
    with pytest.raises(OIDCAuthUnavailable):
        sessions.ensure_fresh(_expire(row))
    assert OIDCSession.objects.get(pk=row.pk).ended_at is None
    assert get_breaker().is_open()
    calls = oidc_op.refresh_calls()
    with pytest.raises(OIDCAuthUnavailable, match='circuit_open'):
        sessions.ensure_fresh(_expire(row))
    assert oidc_op.refresh_calls() == calls  # nobody calls the OP while it is open


def test_issuer_mismatch_invalid_grant_is_not_terminal(row, oidc_op):
    oidc_op.script_token((400, {'error': 'invalid_grant', 'error_description': "Invalid token issuer. Expected 'http://kc'"}))
    with pytest.raises(OIDCAuthUnavailable):
        sessions.ensure_fresh(_expire(row))
    assert OIDCSession.objects.get(pk=row.pk).ended_at is None


def test_response_for_another_subject_is_rejected(row, oidc_op):
    oidc_op.script_token(lambda data: oidc_op.issue_tokens('someone-else'))
    with pytest.raises(OIDCAuthUnavailable):
        sessions.ensure_fresh(_expire(row))
    stored = OIDCSession.objects.get(pk=row.pk)
    assert stored.ended_at is None and stored.version == row.version


def test_refresh_without_new_id_token_keeps_the_previous_one(row, oidc_op):
    data = oidc_op.issue_tokens(row.sub)
    data.pop('id_token')
    oidc_op.script_token(data)
    fresh = sessions.ensure_fresh(_expire(row))
    assert fresh.id_token == row.id_token


def test_no_refresh_inside_a_callers_transaction(row, oidc_op):
    with transaction.atomic():
        assert sessions.ensure_fresh(row).pk == row.pk  # usable: served unchanged
        with pytest.raises(OIDCAuthUnavailable, match='refresh_in_atomic'):
            sessions.ensure_fresh(_expire(row))
    assert oidc_op.refresh_calls() == 0


def test_without_refresh_token_the_sign_in_ends_with_the_access_token(client, user, oidc_op, django_capture_on_commit_callbacks):
    row = create_oidc_session(client, user, access_expires_in=30, refresh=False)
    assert sessions.ensure_fresh(row).pk == row.pk
    with django_capture_on_commit_callbacks(execute=True):
        with pytest.raises(OIDCSessionEnded, match='no_refresh_token'):
            sessions.ensure_fresh(_expire(row))


def test_access_token_for_refreshes_and_reports_ended_sessions(row, oidc_op):
    assert sessions.access_token_for(row.pk, min_ttl=120)
    assert oidc_op.refresh_calls() == 1
    sessions.end_session(row, 'logout')
    with pytest.raises(OIDCSessionEnded):
        sessions.access_token_for(row.pk)
    with pytest.raises(OIDCSessionEnded):
        sessions.access_token_for('not-a-uuid')


def test_access_token_for_serves_a_usable_token_without_queueing_behind_another_refresh(row, oidc_op):
    """Another request of this process is refreshing the sign-in. A consumer asking for a token
    (the dashboard proxy, with min_ttl=60) gets the still-usable one at once instead of queueing
    behind that refresh; only wait=True queues, for at most OIDC_REFRESH_WAIT_SECONDS."""
    in_flight = keyed_lock(('oidc-refresh', str(row.pk)))
    assert in_flight.acquire(blocking=False)
    try:
        with override_settings(OIDC_REFRESH_WAIT_SECONDS=0.5):
            started = time.monotonic()
            served = sessions.access_token_for(row.pk, min_ttl=60)
            served_in = time.monotonic() - started
            started = time.monotonic()
            insisted = sessions.access_token_for(row.pk, min_ttl=60, wait=True)
            insisted_in = time.monotonic() - started
    finally:
        in_flight.release()
    assert served == insisted == row.access_token  # 30 s left: usable, so served
    assert served_in < 0.25
    assert insisted_in >= 0.5
    assert oidc_op.refresh_calls() == 0


def test_requests_queued_behind_a_failed_refresh_do_not_repeat_it(row, oidc_op, monkeypatch):
    """A refresh of the sign-in failed a moment ago. The requests that queued behind it get its
    outcome -- 503 here, as the token has expired -- without calling Keycloak again, so a slow
    Keycloak is not asked once per waiting request. After the gap, the next request tries again."""
    oidc_op.script_token((503, 'Service Unavailable'))
    expired = _expire(row)
    with pytest.raises(OIDCAuthUnavailable):
        sessions.ensure_fresh(expired)
    assert oidc_op.refresh_calls() == 1
    with pytest.raises(OIDCAuthUnavailable):
        sessions.ensure_fresh(expired)  # queued behind the failed attempt
    assert oidc_op.refresh_calls() == 1
    assert not get_breaker().is_open()  # held back by the gap, not by the breaker
    monkeypatch.setattr(sessions, 'REFRESH_RETRY_GAP_SECONDS', 0)
    assert sessions.ensure_fresh(expired).remaining() > 200
    assert oidc_op.refresh_calls() == 2


def test_end_sessions_filters(client, user, oidc_op, django_user_model):
    other = django_user_model.objects.create_user('bob', 'bob@example.test', 'pw-bob-123')
    a = create_oidc_session(client, user, sid='shared')
    b = create_oidc_session(client, other, sid='shared')
    c = create_oidc_session(client, other, sid='own')
    assert sessions.end_sessions(sid='shared', issuer=a.issuer, reason='backchannel_logout') == 2
    assert sessions.end_sessions(sid='shared', reason='again') == 0  # ends once
    assert OIDCSession.objects.get(pk=c.pk).ended_at is None
    assert {OIDCSession.objects.get(pk=pk).end_reason for pk in (a.pk, b.pk)} == {'backchannel_logout'}
    with pytest.raises(ValueError):
        sessions.end_sessions(reason='everything')


def test_max_age_cap(client, user, oidc_op):
    row = create_oidc_session(client, user, access_expires_in=600)
    OIDCSession.objects.filter(pk=row.pk).update(created_at=timezone.now() - timedelta(days=2))
    with override_settings(OIDC_SESSION_MAX_AGE_SECONDS=86400):
        response = client.get('/api/data/', HTTP_SEC_FETCH_MODE='cors')
    assert response.status_code == 401 and response.json()['error'] == 'session_expired'
    assert OIDCSession.objects.get(pk=row.pk).end_reason == 'max_age'


def test_sliding_renewal_is_throttled(client, user, oidc_op):
    row = create_oidc_session(client, user, access_expires_in=600)
    assert client.get('/api/data/', HTTP_SEC_FETCH_MODE='cors').status_code == 200
    assert OIDCSession.objects.get(pk=row.pk).last_seen_at is None  # renewed less than an hour ago
    session = client.session
    session['_oidc_renewed'] = 0
    session.save()
    expire_before = Session.objects.get(session_key=session.session_key).expire_date
    assert client.get('/api/data/', HTTP_SEC_FETCH_MODE='cors').status_code == 200
    assert OIDCSession.objects.get(pk=row.pk).last_seen_at is not None
    assert Session.objects.get(session_key=session.session_key).expire_date >= expire_before
