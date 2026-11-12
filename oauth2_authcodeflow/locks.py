"""In-process keyed locks.

One ``threading.Lock`` per key, created on demand and dropped when the last holder or
waiter releases it. Used as the first stage of the single-flight refresh: requests of one
process queue here without holding a database transaction (and its pooled connection).
"""
import threading
from typing import (
    Dict,
    Hashable,
    List,
    Optional,
)

_guard = threading.Lock()
_registry: Dict[Hashable, List] = {}  # key -> [lock, refcount]


class KeyedLock:
    def __init__(self, key: Hashable) -> None:
        self.key = key
        with _guard:
            entry = _registry.get(key)
            if entry is None:
                entry = [threading.Lock(), 0]
                _registry[key] = entry
            entry[1] += 1
            self._lock: threading.Lock = entry[0]
        self._held = False
        self._closed = False

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        if not blocking:
            ok = self._lock.acquire(False)
        elif timeout is None:
            ok = self._lock.acquire()
        else:
            ok = self._lock.acquire(True, max(0.0, float(timeout)))
        self._held = ok
        return ok

    def release(self) -> None:
        """Release (if held) and drop this handle's reference. Safe to call once in a finally."""
        if self._held:
            self._held = False
            self._lock.release()
        if not self._closed:
            self._closed = True
            with _guard:
                entry = _registry.get(self.key)
                if entry is not None:
                    entry[1] -= 1
                    if entry[1] <= 0:
                        del _registry[self.key]

    def __enter__(self) -> 'KeyedLock':
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def keyed_lock(key: Hashable) -> KeyedLock:
    return KeyedLock(key)


def active_keys() -> int:
    """Number of keys currently registered (tests)."""
    with _guard:
        return len(_registry)


__all__ = ['KeyedLock', 'active_keys', 'keyed_lock']
