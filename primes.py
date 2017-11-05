from mlp import MLP
import numpy as np
import pickle
import time
import threading

def is_prime(n):
  if n == 2 or n == 3: return True
  if n < 2 or n%2 == 0: return False
  if n < 9: return True
  if n%3 == 0: return False
  r = int(n**0.5)
  f = 5
  while f <= r:
    if n%f == 0: return False
    if n%(f+2) == 0: return False
    f +=6
  return True

def softmax(x):
	return np.log(1 + np.exp(x))

def df_softmax(x):
	return 1 / (1 + np.exp(-x))
	
def mine(x):
	return np.cos(x)

def df_mine(x):
	return -np.sin(x)
	
def learn(NN, X, Y):
	start = time.time()
	n = 20
	for i in range(n):
		NN.learn(X,Y)
		NN.__init__(NN.input_layer_neurons, NN.hidden_layer_neurons, NN.output_layer_neurons, f=NN.f, df=NN.df)
	NN.learn(X,Y, threshold=1e-7)
	end = time.time()
	print('Avg time of ' + str(NN.f) + ' is ' + str((end-start)/n))
r = 30
X = np.array([[i/r] for i in range(r)])
Y = np.array([[int(is_prime(i))] for i in range(r)])
# X = np.array([[0, 1],[1, 0],[0, 0],[1, 1]]) # or gate
# Y = [1,1,0,1]
NN_softmax = MLP(1, 100, 1, f=softmax, df=df_softmax)
NN_sigmoid = MLP(1, 2, 1)
NN_mine = MLP(2, 2, 1, f=mine, df=df_mine)
NN_softmax.learn(X,Y)
# learn(NN_mine, X, Y)

x = np.array([0.17])
f_o = NN_softmax.feed_forward(x)
print(np.round(f_o))
# threading.Thread(target=NN_mine.learn, args=[X,Y]).start()
# threading.Thread(target=NN_softmax.learn, args=[X,Y]).start()
# threading.Thread(target=NN_sigmoid.learn, args=[X,Y]).start()
