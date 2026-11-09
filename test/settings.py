DEBUG = False
INSTALLED_APPS = (
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'rest_framework',
    'channels',
    'oauth2_authcodeflow',
)
ROOT_URLCONF = 'test.urls'
SECRET_KEY = 'test-secret-key-for-oauth2-authcodeflow'
MIDDLEWARE = [
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'oauth2_authcodeflow.middleware.BearerAuthMiddleware',
    'oauth2_authcodeflow.middleware.OIDCSessionMiddleware',
    'oauth2_authcodeflow.middleware.LoginRequiredMiddleware',
]
AUTHENTICATION_BACKENDS = [
    'oauth2_authcodeflow.auth.AuthenticationBackend',
    'django.contrib.auth.backends.ModelBackend',
]
TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': [
        'django.template.context_processors.request',
        'django.contrib.auth.context_processors.auth',
        'django.contrib.messages.context_processors.messages',
    ]},
}]
USE_TZ = True
DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}}
CACHES = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}
CHANNEL_LAYERS = {'default': {'BACKEND': 'channels.layers.InMemoryChannelLayer'}}
SESSION_COOKIE_AGE = 31 * 86400
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'oauth2_authcodeflow.contrib.drf.OIDCBearerAuthentication',
        'oauth2_authcodeflow.contrib.drf.OIDCSessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': ['rest_framework.permissions.IsAuthenticated'],
    'EXCEPTION_HANDLER': 'oauth2_authcodeflow.contrib.drf.oidc_exception_handler',
}
OIDC_RP_CLIENT_ID = 'test-client'
OIDC_RP_CLIENT_SECRET = 'test-secret'
OIDC_OP_DISCOVERY_DOCUMENT_URL = 'https://op.example.test/realms/test/.well-known/openid-configuration'
OIDC_OP_ISSUER = 'https://op.example.test/realms/test'
OIDC_LOGIN_REQUIRED_EXEMPT_URL_PATTERNS = [r'^/public/', r'^/health$', r'^/admin/']
OIDC_SESSION_EXEMPT_URL_PATTERNS = [r'^/health$']
SILENCED_SYSTEM_CHECKS = ['oauth2_authcodeflow.W001']
