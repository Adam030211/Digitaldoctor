from django.urls import path
from . import views

urlpatterns = [
    path("", views.generate_text, name="generate_text"),
    path("about/",views.about, name="about" ),
    path("home/",views.home, name="home" )

]