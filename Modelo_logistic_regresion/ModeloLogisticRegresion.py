from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import chess.pgn as chessy
import re

# import sys
# np.set_printoptions(threshold=sys.maxsize)

pgn = open("prueba.pgn")
pgn2 = open("prueba2.pgn")
first_game = chessy.read_game(pgn)
second_game = chessy.read_game(pgn2)

board = first_game.board()
board2 = second_game.board()

X = []
Y = []

for move in first_game.mainline_moves():
    print(board)
    print(str(board.fen()))
    Y.append(str(move))
    X.append(str(board.fen()).split(" "))
    board.push(move)
    print(move)
    print('\n')

for move in second_game.mainline_moves():
    Y.append(str(move))
    X.append(str(board2.fen()).split(" "))
    board2.push(move)

Xtis = np.array(X)
Xt = Xtis[:,0]
Xtrain = Xt[1::2]
ytrain = Y[1::2]

print(len(ytrain))
print(len(Xtrain))
Xtrain = np.array(Xtrain)
print(Xtrain)

def convertir_a_numeros(cadena):
    cadena = cadena.replace("8", "99.99.99.99.99.99.99.99.")
    cadena = cadena.replace("7", "99.99.99.99.99.99.99.")
    cadena = cadena.replace("6", "99.99.99.99.99.99.")
    cadena = cadena.replace("5", "99.99.99.99.99.")
    cadena = cadena.replace("4", "99.99.99.99.")
    cadena = cadena.replace("3", "99.99.99.")
    cadena = cadena.replace("2", "99.99.")
    cadena = cadena.replace("1", "99.")

    cadena = cadena.replace('r', '10.')
    cadena = cadena.replace('n', '11.')
    cadena = cadena.replace('b', '12.')
    cadena = cadena.replace('q', '13.')
    cadena = cadena.replace('k', '14.')
    cadena = cadena.replace('p', '15.')

    cadena = cadena.replace('R', '16.')
    cadena = cadena.replace('N', '17.')
    cadena = cadena.replace('B', '18.')
    cadena = cadena.replace('Q', '19.')
    cadena = cadena.replace('K', '20.')
    cadena = cadena.replace('P', '21.')

    return cadena

Xtrain_v = []

for i in range(len(Xtrain)):

    x = Xtrain[i].split("/")
    cadena1 = convertir_a_numeros(x[0])
    cadena2 = convertir_a_numeros(x[1])
    cadena3 = convertir_a_numeros(x[2])
    cadena4 = convertir_a_numeros(x[3])
    cadena5 = convertir_a_numeros(x[4])
    cadena6 = convertir_a_numeros(x[5])
    cadena7 = convertir_a_numeros(x[6])
    cadena8 = convertir_a_numeros(x[7])

    cadena_unida = cadena1 + cadena2 + cadena3 + cadena4 + cadena5 + cadena6 + cadena7 + cadena8
    Xtrain_v.append(cadena_unida)
print(Xtrain_v)

matriz = []
for i in range(len(Xtrain_v)):
    vector = Xtrain_v[i].split(".")
    vector.pop()
    matriz.append(vector)

matriz = np.array(matriz)
print(matriz)

print(matriz.shape)

matriz = matriz.astype(np.float)
matriz

for i in range(len(ytrain)):
    ytrain[i] = ytrain[i].replace("a", "1")
    ytrain[i] = ytrain[i].replace("b", "2")
    ytrain[i] = ytrain[i].replace("c", "3")
    ytrain[i] = ytrain[i].replace("d", "4")
    ytrain[i] = ytrain[i].replace("e", "5")
    ytrain[i] = ytrain[i].replace("f", "6")
    ytrain[i] = ytrain[i].replace("g", "7")
    ytrain[i] = ytrain[i].replace("h", "8")

print(ytrain)

ytrain = np.array(ytrain)
ytrain = ytrain.astype(np.float)
print(ytrain)

X_train, X_test, y_train, y_test =train_test_split(matriz,ytrain,test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(len(y_train))
print(len(y_test))

def remocion_train_test(X_train, X_test):
    media_train = X_train.mean(axis = 0)
    std_train = X_train.std(axis = 0)

    X_train_normalizado = (X_train - media_train)/std_train
    X_test_normalizado = (X_test - media_train)/std_train

    return X_train_normalizado , X_test_normalizado

X_train_normalizado, X_test_normalizado = remocion_train_test(X_train, X_test)

X_train_normalizado = np.nan_to_num(X_train_normalizado)
X_test_normalizado = np.nan_to_num(X_test_normalizado)

clasificador1 = LogisticRegression(C=250.0, random_state = 0, solver='lbfgs', multi_class='multinomial', max_iter=10000)
clasificador2 = LogisticRegression(C=500.0, random_state = 0, solver='lbfgs', multi_class='multinomial')
clasificador3 = LogisticRegression(C=750.0, random_state = 0, solver='lbfgs', multi_class='multinomial')
clasificador4 = LogisticRegression(C=1000.0, random_state = 0, solver='lbfgs', multi_class='multinomial')
clasificador5 = LogisticRegression(C=1250.0, random_state = 0, solver='lbfgs', multi_class='multinomial')

def entrenador_remocion(clasificador):
    clasificador.fit(X_train_normalizado,y_train)
    y_pred = clasificador.predict(X_test_normalizado)
    print('Las muestras mal clasificadas fueron %d' % (y_test != y_pred).sum())
    print(y_pred)
    print(y_test)
    print('///////////////////////////////////')

entrenador_remocion(clasificador1)
# entrenador_remocion(clasificador2)
# entrenador_remocion(clasificador3)
# entrenador_remocion(clasificador4)
# entrenador_remocion(clasificador5)
