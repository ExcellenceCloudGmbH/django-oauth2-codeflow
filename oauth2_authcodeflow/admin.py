from django.contrib import (
    admin,
    messages,
)

from . import sessions
from .models import OIDCSession


@admin.register(OIDCSession)
class OIDCSessionAdmin(admin.ModelAdmin):
    """Read-only list of sign-ins; tokens are never shown. Ending is an action."""
    list_display = ('id', 'user', 'sub', 'sid', 'client_id', 'created_at', 'last_refresh_at', 'last_seen_at', 'ended_at', 'end_reason')
    list_filter = ('end_reason', ('ended_at', admin.EmptyFieldListFilter), 'migrated_from_legacy')
    search_fields = ('sub', 'sid', 'client_id')
    exclude = ('access_token_enc', 'refresh_token_enc', 'id_token_enc')
    actions = ['end_selected']
    ordering = ('-created_at',)

    def has_add_permission(self, request) -> bool:
        return False

    def has_change_permission(self, request, obj=None) -> bool:
        return False

    @admin.action(description='End the selected sign-ins', permissions=['delete'])
    def end_selected(self, request, queryset) -> None:
        ended = sum(1 for row in queryset if sessions.end_session(row, 'admin'))
        self.message_user(request, f'{ended} sign-in(s) ended.', messages.SUCCESS)
