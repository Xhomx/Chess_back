from django.shortcuts import render
from rest_framework import generics
from modelo1.models import Jugada, Tablero
from modelo1.serializers import JugadaSerializer, TableroSerializer

# Create your views here.

class JugadaList(generics.ListCreateAPIView):
    queryset = Jugada.objects.all()
    serializer_class = JugadaSerializer

class JugadaDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Jugada.objects.all()
    serializer_class = JugadaSerializer

class TableroList(generics.ListCreateAPIView):
    queryset = Tablero.objects.all()
    serializer_class = TableroSerializer

class TableroDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Tablero.objects.all()
    serializer_class = TableroSerializer
