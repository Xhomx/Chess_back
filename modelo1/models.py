from django.db import models

# Create your models here.

class Jugada(models.Model):
    movimiento = models.CharField(max_length=50)

class Tablero(models.Model):
    posicion = models.CharField(max_length=255)
