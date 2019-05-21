from django.shortcuts import render
from rest_framework import generics
from modelo1.models import Jugada, Tablero
from modelo1.serializers import JugadaSerializer, TableroSerializer
from joblib import load
import numpy as np
import threading
import time

# Create your views here.

class JugadaList(generics.ListCreateAPIView):
	def calcularJugada():

		def numberToLetter(number):
			if int(number) == 1:
				return 'a'
			if int(number) == 2:
				return 'b'
			if int(number) == 3:
				return 'c'
			if int(number) == 4:
				return 'd'
			if int(number) == 5:
				return 'e'
			if int(number) == 6:
				return 'f'
			if int(number) == 7:
				return 'g'
			if int(number) == 8:
				return 'h'

		while True:
			model = load('C:/Users/David Acevedo/Desktop/chess_back/Chess_back_/modelo1/modelo.joblib')
			vectorTablero = np.zeros((1, 64))
			querySetTablero = Tablero.objects.all()
			stringTablero = str(querySetTablero.values_list('posicion', flat = True))
			stringTablero = stringTablero[12: -1]
			stringTablero = stringTablero.split(',')
			vectorTablero = np.zeros((1, 64))
			for i in range(0, 63):
				vectorTablero[0, i] = stringTablero[i]
			stringConvertir = str(model.predict(vectorTablero))
			stringConvertir = stringConvertir[1:-2]
			stringEdit = numberToLetter(stringConvertir[0]) + stringConvertir[1] + numberToLetter(stringConvertir[2]) + stringConvertir[3]
			jugada = Jugada(movimiento = stringEdit)
			jugada.save()
			#queryset = Jugada.objects.all()
			#serializer_class = JugadaSerializer
			time.sleep(5)

	thread = threading.Thread(target= calcularJugada)
	thread.start()
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
