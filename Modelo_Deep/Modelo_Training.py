#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import chess.pgn as chessy


# In[2]:


pgn = open("prueba.pgn")
first_game = chessy.read_game(pgn)

board = first_game.board()

X = []
Y = []
t = []

for move in first_game.mainline_moves():
    print(board)
    print(str(board.fen()))
    Y.append(str(move))
    X.append(str(board.fen()).split(" "))
    board.push(move)
    print(board.peek())
    print(move)
    print('\n')
    print(board)


# In[3]:


X = np.array(X)
print(X[:,0])
Xt = np.array(X[:,0])
X = Xt[1::2]


# In[4]:


def remove_Numbers(cadena):
    cadena = cadena.replace("8", "00000000")
    cadena = cadena.replace("7", "0000000")
    cadena = cadena.replace("6", "000000")
    cadena = cadena.replace("5", "00000")
    cadena = cadena.replace("4", "0000")
    cadena = cadena.replace("3", "000")
    cadena = cadena.replace("2", "00")
    cadena = cadena.replace("1", "0")
    
    return cadena


# In[5]:


def toNumber(cadena):
    cadena = cadena.replace('r', '10')
    cadena = cadena.replace('n', '11')
    cadena = cadena.replace('b', '12')
    cadena = cadena.replace('q', '13')
    cadena = cadena.replace('k', '14')
    cadena = cadena.replace('p', '15')

    cadena = cadena.replace('R', '16')
    cadena = cadena.replace('N', '17')
    cadena = cadena.replace('B', '18')
    cadena = cadena.replace('Q', '19')
    cadena = cadena.replace('K', '20')
    cadena = cadena.replace('P', '21')
    
    return cadena


# In[6]:


def matrixOf(x):
    cadena = ""
    Xnew = np.zeros((len(w), 64))
    for i in range(len(x)):
        cadena = str(x[i])[2:-2]
        for j in range(0,64):
            Xnew[i,j] = int(toNumber(cadena[j]))
    return Xnew


# In[7]:


x_vals = []

for i in range(len(X)):
    
    x = X[i].split("/")
    cadena1 = remove_Numbers(x[0])
    cadena2 = remove_Numbers(x[1])
    cadena3 = remove_Numbers(x[2])
    cadena4 = remove_Numbers(x[3])
    cadena5 = remove_Numbers(x[4])
    cadena6 = remove_Numbers(x[5])
    cadena7 = remove_Numbers(x[6])
    cadena8 = remove_Numbers(x[7])
    
    cadena_unida = cadena1 + cadena2 + cadena3 + cadena4 + cadena5 + cadena6 + cadena7 + cadena8
#     x_vals.append(float(cadena_unida))
    x_vals.append(cadena_unida)
print(x_vals)


# In[8]:


count = 0
b = []
w = []
for i in x_vals:
    b.append(i)
    
w.append(b)
    
b = np.array(b)
# b = b.reshape(8,8)
# print(b)
w = np.transpose(w)
print(w)


# In[9]:


Xnew = matrixOf(w)
print(Xnew)


# In[10]:


#Xtrain = Xt[1::2]
Xtrain = Xnew
ytrain = Y[1::2]
ytrain = np.array(ytrain)
print(ytrain)


# In[16]:


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


# In[11]:


print(len(ytrain))
print(len(Xtrain))


# In[12]:


modeloJugada = tf.keras.models.Sequential()


# In[21]:


modeloJugada.add(tf.keras.layers.Dense(8888, input_shape = (64,), activation = tf.nn.relu))
modeloJugada.add(tf.keras.layers.Dense(4096, activation = tf.nn.relu))
modeloJugada.add(tf.keras.layers.Dense(4096, activation = tf.nn.relu))
modeloJugada.add(tf.keras.layers.Dense(4096, activation = tf.nn.relu))
modeloJugada.add(tf.keras.layers.Dense(8888))


# In[22]:


modeloJugada.compile(optimizer = tf.keras.optimizers.Adadelta(0.001),
                    loss = 'sparse_categorical_crossentropy')


# In[ ]:


modeloJugada.fit(Xtrain, ytrain, epochs = 500)


# In[ ]:





# In[ ]:




