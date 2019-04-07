from django.db import models

# Create your models here.

class Jugada(models.Model):
    posicion = models.CharField(max_length=255)
    movimiento = models.CharField(max_length=50)
