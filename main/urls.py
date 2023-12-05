from django.urls import path

from .views import index, serp

app_name = 'main'

urlpatterns = [
    path("", index, name="index"),
    path("search", serp, name="search"),
]