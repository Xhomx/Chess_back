from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^jugada$', views.JugadaList.as_view()),
    url(r'jugada/(?P<pk>[0-9]+)$', views.JugadaDetail.as_view()),
    url(r'^tablero$', views.TableroList.as_view()),
    url(r'tablero/(?P<pk>[0-9]+)$', views.TableroDetail.as_view()),
]
