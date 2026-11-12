from django.apps import AppConfig


class Oauth2AuthcodeflowConfig(AppConfig):
    name = 'oauth2_authcodeflow'
    verbose_name = 'OIDC authentication'
    # pinned so that projects with DEFAULT_AUTO_FIELD = BigAutoField never generate a
    # migration for BlacklistedToken.id inside site-packages
    default_auto_field = 'django.db.models.AutoField'

    def ready(self) -> None:
        from . import checks  # noqa: F401  (registers the system checks)
        from .conf import settings
        settings.warn_removed()
