from rest_framework import serializers
from modelo1.models import Jugada, Tablero

class JugadaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Jugada
        fields = ('id', 'movimiento')

class TableroSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tablero
        fields = ('id', 'posicion')
