from django.contrib import admin
from django.urls import (
    include,
    path,
)

from . import views

urlpatterns = [
    path('oidc/', include('oauth2_authcodeflow.urls')),
    path('admin/', admin.site.urls),
    path('protected/', views.protected),
    path('public/', views.public),
    path('public/decorated/', views.decorated),
    path('health', views.health),
    path('api/data/', views.api_data),
    path('api/fresh/', views.api_fresh),
    path('api/raise/', views.api_raise),
    path('api/drf/', views.DRFData.as_view()),
    path('api/drf-denied/', views.DRFDenied.as_view()),
    path('api/drf-raise/', views.DRFRaise.as_view()),
]
