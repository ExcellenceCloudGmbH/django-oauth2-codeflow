"""Django Channels integration.

- ``OIDCSessionASGIMiddleware`` (inside ``AuthMiddlewareStack``) resolves the active sign-in
  of the WebSocket's Django session into ``scope['oidc_session']`` (``None`` when there is
  none). It never refreshes tokens.
- ``OIDCSessionConsumerMixin`` (async consumers): call ``await self.oidc_connect()`` first in
  ``connect()``. Anonymous sockets are accepted and closed with **4401** (closing before
  ``accept()`` reaches the browser as 1006). Authenticated sockets join
  ``group_name(session_id)``, are closed with 4401 when the sign-in ends (signal → group
  message) and re-validated every ``oidc_revalidate_seconds``.
"""
import asyncio
import hashlib
import random
from typing import (
    Any,
    Dict,
    Optional,
)

from asgiref.sync import async_to_sync
from channels.db import database_sync_to_async
from channels.layers import get_channel_layer
from channels.middleware import BaseMiddleware

from ..conf import settings

CLOSE_UNAUTHENTICATED = 4401
CLOSE_FORBIDDEN = 4403
CLOSE_AUTH_UNAVAILABLE = 4503


def group_name(session_id: Any) -> str:
    """Channel-layer group of a sign-in (ASCII, < 100 characters)."""
    return 'oidc-session.' + hashlib.sha256(str(session_id).encode('utf-8')).hexdigest()[:32]


def _resolve(scope: Dict[str, Any]) -> Optional[Dict[str, str]]:
    from ..models import OIDCSession
    session = scope.get('session')
    if session is None:
        return None
    key = session.get(settings.OIDC_SESSION_KEY)
    if not key:
        return None
    try:
        row = OIDCSession.objects.filter(pk=key, ended_at__isnull=True).first()
    except Exception:
        return None
    user = scope.get('user')
    user_pk = getattr(user, 'pk', None) if user is not None and getattr(user, 'is_authenticated', False) else None
    if row is None or user_pk is None or user_pk != row.user_id:
        return None
    return {'session_id': str(row.pk), 'sub': row.sub, 'sid': row.sid}


def _is_active(session_id: str) -> bool:
    from ..models import OIDCSession
    return OIDCSession.objects.filter(pk=session_id, ended_at__isnull=True).exists()


class OIDCSessionASGIMiddleware(BaseMiddleware):
    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> Any:
        scope = dict(scope)
        if scope.get('type') == 'websocket':
            scope['oidc_session'] = await database_sync_to_async(_resolve)(scope)
        return await super().__call__(scope, receive, send)


class OIDCSessionConsumerMixin:
    """For ``AsyncWebsocketConsumer`` / ``AsyncJsonWebsocketConsumer``."""
    oidc_revalidate_seconds = 300
    oidc_close_code = CLOSE_UNAUTHENTICATED

    async def oidc_connect(self) -> bool:
        info = self.scope.get('oidc_session')  # type: ignore[attr-defined]
        if not info:
            await self.accept()  # type: ignore[attr-defined]
            await self.close(code=self.oidc_close_code)  # type: ignore[attr-defined]
            return False
        self._oidc_session_id = info['session_id']
        self._oidc_group = group_name(info['session_id'])
        layer = getattr(self, 'channel_layer', None)
        if layer is not None:
            await layer.group_add(self._oidc_group, self.channel_name)  # type: ignore[attr-defined]
        if self.oidc_revalidate_seconds:
            self._oidc_task = asyncio.ensure_future(self._oidc_revalidate())
        return True

    async def oidc_disconnect(self) -> None:
        task = getattr(self, '_oidc_task', None)
        if task is not None:
            task.cancel()
        group = getattr(self, '_oidc_group', None)
        layer = getattr(self, 'channel_layer', None)
        if group and layer is not None:
            await layer.group_discard(group, self.channel_name)  # type: ignore[attr-defined]

    async def oidc_session_ended(self, event: Dict[str, Any]) -> None:
        """Group message ``{'type': 'oidc.session.ended'}``."""
        await self.close(code=self.oidc_close_code)  # type: ignore[attr-defined]

    async def _oidc_revalidate(self) -> None:
        base = float(self.oidc_revalidate_seconds)
        try:
            while True:
                await asyncio.sleep(base + random.uniform(-0.1, 0.1) * base)
                if not await database_sync_to_async(_is_active)(self._oidc_session_id):
                    await self.close(code=self.oidc_close_code)  # type: ignore[attr-defined]
                    return
        except asyncio.CancelledError:
            pass


class SyncOIDCSessionConsumerMixin:
    """For synchronous ``WebsocketConsumer`` / ``JsonWebsocketConsumer`` (no periodic
    re-validation; sign-in ends are delivered through the group message)."""
    oidc_close_code = CLOSE_UNAUTHENTICATED

    def oidc_connect(self) -> bool:
        info = self.scope.get('oidc_session')  # type: ignore[attr-defined]
        if not info:
            self.accept()  # type: ignore[attr-defined]
            self.close(code=self.oidc_close_code)  # type: ignore[attr-defined]
            return False
        self._oidc_group = group_name(info['session_id'])
        layer = getattr(self, 'channel_layer', None)
        if layer is not None:
            async_to_sync(layer.group_add)(self._oidc_group, self.channel_name)  # type: ignore[attr-defined]
        return True

    def oidc_disconnect(self) -> None:
        group = getattr(self, '_oidc_group', None)
        layer = getattr(self, 'channel_layer', None)
        if group and layer is not None:
            async_to_sync(layer.group_discard)(group, self.channel_name)  # type: ignore[attr-defined]

    def oidc_session_ended(self, event: Dict[str, Any]) -> None:
        self.close(code=self.oidc_close_code)  # type: ignore[attr-defined]


def notify_session_ended(row: Any) -> None:
    """Close the sockets of an ended sign-in (called by ``sessions.after_end``)."""
    mode = (settings.OIDC_CHANNELS_NOTIFY or 'auto').lower()
    if mode == 'off':
        return
    layer = get_channel_layer()
    if layer is None:
        return
    async_to_sync(layer.group_send)(group_name(row.pk), {'type': 'oidc.session.ended'})


__all__ = [
    'CLOSE_AUTH_UNAVAILABLE',
    'CLOSE_FORBIDDEN',
    'CLOSE_UNAUTHENTICATED',
    'OIDCSessionASGIMiddleware',
    'OIDCSessionConsumerMixin',
    'SyncOIDCSessionConsumerMixin',
    'group_name',
    'notify_session_ended',
]
