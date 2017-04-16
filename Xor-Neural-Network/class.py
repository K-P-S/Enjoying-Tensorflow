#two layer neural network
#predict XOR value
#credit: youtube, Sirajology

#dependencies
import numpy as np
import time

#variables
n_hidden = 10
n_in = 10
#outputs
n_out = 10
#sample data
n_sample = 300

#hyperparameters
learning_rate = 0.01
momentum = 0.9

#(non deterministic) seeding generates the same random numbers each time code runs
np,.random.seed(0)

#define activation function (sigmoid)
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
#define activation function (tangent)
def tanh_prime(x):
  return 1 - np.tanh(x)**2

#training function
#takes five parameters: input data, transpose, layer 1, layer 2, bias 1, bias 2
def train(x, t, V, W, bv, bw):
  
  #forard propogation; matrix multiply + biases
  A = np.dot(x, V) + bv
  Z = np.tanh(A) #first activation func
  
  B = np.dot(Z, W) + bw
  Y = sigmoid(B)
  
  #backward propogation
  Ew = Y - t
  Ev = tanh_prime(A) + np.dot(W, Ew) 
  
  #predicting loss
  dW = np.outer(Z, Ew)
  dV = np.outer(x, Ev)
  #cross entropy
  loss = -np.mean(t * np.log(Y) + (1 - t) * np.log(1-Y))
  
  return loss, (dV, dW, Ev, Ew)


def predict(x, V, W, bv, bw):
  A = np.dot(x, V) + bv
  B = np.dot(np.tanh(A), W) + bw
  return (sigmoid(B) > 0.5).atype(int) #return 1 if value > .5, 0 ow

#create layers
V = np.random.normal(scale = 0.1, size = (n_in, n_hidden))
W = np.random.normal(scale = 0.1, size = (n_hidden, n_out))
#initialize biases
bv = np.zeroes(n_hidden)
bw = np.zeroes(n_out)

parameters = [V, W, bv, bw]
#generate data
X = np.rndom.binomial(1, 0.5, (n_sample, n_in))
T = X ^ 1

#training
for epoch in range(100):
  err = []
  update = [0]+len(parameters)
  #initialize time
  t0 = time.clock()
  #for each data point, update weights
  for i in range(X.shape[0]):
    loss,grad = train(X[i], T[i], *parameters)
    #update loss
    for j in range(len(parameters)):
      parameters[j] -= update[j]
      
    for j in range(len(parameters)):
      update[j] = learning_rate * grad[j] + momentum * update[j]
    #append error with loss
    err.append(loss)
print('Epoch: %d, Loss: %.8f, Time: %fs'%(epoch, np.mean(err), time.clock()-t0))

#try predicting
X = np.random.binomial(1, 0.5, n_in)
print('XOR prediction')
print(x)
print(predict(x, *parameters))
  
  
  
