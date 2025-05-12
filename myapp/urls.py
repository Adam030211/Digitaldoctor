from django.urls import path
from . import views

urlpatterns = [
    path("generate_text", views.generate_text, name="generate_text"),
    path("", views.test, name="test")

]