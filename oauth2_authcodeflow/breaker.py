"""Per-process circuit breaker for calls to the OP.

closed → (``failures`` transient failures within ``window`` s, or one configuration error)
→ open → (after ``cooldown`` / ``config_cooldown``) → half-open: one probe is let through;
success closes the breaker, failure opens it again. While it is open nobody calls the OP:
requests whose access token is still valid keep working, the others get a 503.
"""
import logging
import math
import threading
from collections import deque
from time import monotonic
from typing import (
    Deque,
    Dict,
    Optional,
)

from django.core.signals import setting_changed

from . import events
from .conf import settings

CLOSED = 'closed'
OPEN = 'open'
HALF_OPEN = 'half_open'


class CircuitBreaker:
    def __init__(self, name: str, *, failures: int = 5, window: float = 30, cooldown: float = 15,
                 config_cooldown: float = 300) -> None:
        self.name = name
        self.failures = failures
        self.window = window
        self.cooldown = cooldown
        self.config_cooldown = config_cooldown
        self._lock = threading.Lock()
        self._state = CLOSED
        self._recent: Deque[float] = deque()
        self._open_until = 0.0
        self._probe_in_flight = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._current_state(monotonic())

    def _current_state(self, now: float) -> str:
        if self._state == OPEN and now >= self._open_until:
            return HALF_OPEN if not self._probe_in_flight else OPEN
        return self._state

    def is_open(self) -> bool:
        """True while calls must not be made and no probe is due."""
        with self._lock:
            now = monotonic()
            if self._state == OPEN:
                return now < self._open_until or self._probe_in_flight
            if self._state == HALF_OPEN:
                return self._probe_in_flight
            return False

    def try_acquire(self) -> bool:
        """May a call be made now? In half-open state only one probe is let through."""
        with self._lock:
            now = monotonic()
            if self._state == CLOSED:
                return True
            if self._probe_in_flight:
                return False
            if self._state == OPEN and now < self._open_until:
                return False
            self._state = HALF_OPEN
            self._probe_in_flight = True
            return True

    def success(self) -> None:
        with self._lock:
            was = self._state
            self._state = CLOSED
            self._recent.clear()
            self._probe_in_flight = False
        if was != CLOSED:
            events.emit('oidc.circuit.closed', logging.INFO, breaker=self.name)

    def failure(self, category: str) -> None:
        opened = False
        with self._lock:
            now = monotonic()
            self._probe_in_flight = False
            if category == 'config':
                self._open(now, self.config_cooldown)
                opened = True
            else:
                self._recent.append(now)
                while self._recent and now - self._recent[0] > self.window:
                    self._recent.popleft()
                if self._state == HALF_OPEN or len(self._recent) >= self.failures:
                    self._open(now, self.cooldown)
                    opened = True
        if opened:
            events.emit('oidc.circuit.opened', logging.ERROR, breaker=self.name, category=category,
                        retry_after=self.retry_after())

    def release_probe(self) -> None:
        """Give back a probe that was acquired but not used."""
        with self._lock:
            self._probe_in_flight = False

    def _open(self, now: float, seconds: float) -> None:
        self._state = OPEN
        self._open_until = now + seconds
        self._recent.clear()

    def retry_after(self) -> int:
        with self._lock:
            remaining = self._open_until - monotonic()
        return max(1, int(math.ceil(remaining))) if remaining > 0 else max(1, int(settings.OIDC_RETRY_AFTER_SECONDS))


_lock = threading.Lock()
_breakers: Dict[str, CircuitBreaker] = {}


def get_breaker(name: str = 'token_endpoint') -> CircuitBreaker:
    breaker = _breakers.get(name)
    if breaker is None:
        with _lock:
            breaker = _breakers.get(name)
            if breaker is None:
                conf = dict(settings.OIDC_CIRCUIT_BREAKER or {})
                breaker = CircuitBreaker(
                    name,
                    failures=int(conf.get('failures', 5)),
                    window=float(conf.get('window', 30)),
                    cooldown=float(conf.get('cooldown', 15)),
                    config_cooldown=float(conf.get('config_cooldown', 300)),
                )
                _breakers[name] = breaker
    return breaker


def reset_breakers() -> None:
    with _lock:
        _breakers.clear()


def _reset(**kwargs: Optional[str]) -> None:
    if kwargs.get('setting') == 'OIDC_CIRCUIT_BREAKER':
        reset_breakers()


setting_changed.connect(_reset)

__all__ = ['CLOSED', 'HALF_OPEN', 'OPEN', 'CircuitBreaker', 'get_breaker', 'reset_breakers']
