import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
import chess.pgn as chessy

session = tf.Session()

pgn = open("../../../prueba.pgn")
pgn2 = open("../../../prueba2.pgn")
first_game = chessy.read_game(pgn)
second_game = chessy.read_game(pgn2)

board = first_game.board()
board2 = second_game.board()

X = []
Y = []

z = []

for move in first_game.mainline_moves():
    print(board)
    print(str(board.fen()))
    Y.append(str(move))
    X.append(str(board.fen()).split(" "))
    board.push(move)
    print(move)
    print('\n')

# for move in second_game.mainline_moves():
#     Y.append(str(move))
#     X.append(str(board2.fen()).split(" "))
#     board2.push(move)

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
ytrain

train_idx = np.random.choice(len(matriz), size=round(len(matriz)*0.8), replace = False)
test_idx = np.array(list(set(range(len(matriz)))-set(train_idx)))

x_vals_train = matriz[train_idx]
y_vals_train = ytrain[train_idx]

x_vals_test = matriz[test_idx]
y_vals_test = ytrain[test_idx]

print(x_vals_train.shape)
print(x_vals_test.shape)
print(len(y_vals_train))
print(len(y_vals_test))

print(y_vals_train)
print(y_vals_test)

def normalize_cols(m, col_min = np.array([None]), col_max=np.array([None])):
    if not col_min[0]:
        col_min = m.min(axis=0)
    if not col_max[0]:
        col_max = m.max(axis=0)
    return (m-col_min)/(col_max-col_min), col_min, col_max

x_vals_train, train_min, train_max = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test, test_min, test_max = np.nan_to_num(normalize_cols(x_vals_test, col_min=train_min, col_max=train_max))

x_vals_train = np.nan_to_num(x_vals_train)
x_vals_test = np.nan_to_num(x_vals_test)

batch_size = 10
x_data = tf.placeholder(shape=[None,64], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[64,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
y_pred = tf.add(tf.matmul(x_data, A), b)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels= y_target))
init = tf.global_variables_initializer()
session.run(init)
my_optim = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_step = my_optim.minimize(loss)

clasification_lr = tf.round(tf.sigmoid(y_pred))
correct_classification = tf.cast(tf.equal(clasification_lr, y_target), tf.float32)
accuracy = tf.reduce_mean(correct_classification)

loss_vec = []
train_acc = []
test_acc = []
for i in range(2000):
    rand_idx = np.random.choice(len(x_vals_train), size= batch_size)
    rand_x = x_vals_train[rand_idx]
    rand_y = np.transpose([y_vals_train[rand_idx]])
    session.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = session.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    temp_acc_train = session.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
    train_acc.append(temp_acc_train)
    temp_acc_test = session.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_acc.append(temp_acc_test)

#     if (i+1)% 100 == 0:
#         print("Paso #"+ str(i+1)+ ", A="+str(session.run(A))+", b="+str(session.run(b))+
#              ", loss = "+str(temp_loss))
    if (i+1)% 100 == 0:
        print("Loss = "+str(temp_loss))

plt.plot(loss_vec, 'k-')
plt.title("Funcion de perdidas por generacion")
plt.xlabel("Iteracion")
plt.ylabel("Perdida")
plt.show()

plt.plot(train_acc, 'b-', label= "Precision conjunto entrenamiento")
plt.plot(test_acc, 'r--', label= "Precision conjunto de pruebas")
plt.title("Precision en entrenamiento y testing")
plt.xlabel("Iteracion")
plt.ylabel("Precision (proporcion de clasificacion OK)")
plt.legend(loc= "lower right")
plt.show()
