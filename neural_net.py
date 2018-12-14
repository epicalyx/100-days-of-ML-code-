#!/usr/bin/env python
# coding: utf-8

# # Neural Network as an XOR gate

# In[6]:


'''
Useful Resource and Credit : https://www.youtube.com/watch?v=vcZub77WvFA&index=6&list=PL2-dafEMk2A7mfQDsEcmxxtxgFEZg0bW-
'''

#importing library dependencies
import numpy as np
import time


# In[7]:


#variables
n_hidden_layers=10
n_inputs=10
#no of output variables
n_outputs=10
n_samples=300

#hyperparameters
learning_rate= 0.01
momentum= 0.9


np.random.seed(0)

#activation functions

#for output layer
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

#for hidden layer
def tanh_prime(x):
    return 1 - np.tanh(x)**2


#training model
def train(x,t,V,W,bv,bw):
    
    # forward 
    A = np.dot(x,V) + bv
    Z = np.tanh(A)
    
    B = np.dot(Z,W) + bw
    Y = sigmoid(B)
    
    
    # backward 
    Ew = Y-t
    Ev = tanh_prime(A)*np.dot(W,Ew)
    
    dW = np.outer(Z,Ew)
    dV = np.outer(x,Ev)
    
    #calculate loss
    loss = -np.mean(t*np.log(Y) + (1-t)*np.log(1-Y))
    
    
    return loss,(dV,dW,Ev,Ew)

#prediction
def predict(x,V,W,bv,bw):
    A = np.dot(x,V) + bv
    B = np.dot(np.tanh(A),W) + bw
    
    return (sigmoid(B)>0.5).astype(int)


#Setup initial parameters as initialization is important for first order methods

V = np.random.normal(scale=0.1,size=(n_inputs,n_hidden_layers))
W = np.random.normal(scale=0.1,size=(n_hidden_layers,n_outputs))

#initializing biases
bv = np.zeros(n_hidden_layers)
bw = np.zeros(n_outputs)

params = [V,W,bv,bw]

#Generate some data

X = np.random.binomial(1,0.5,(n_samples,n_inputs))
T = X^1

#Train
for epoch in range(100):
    err = []
    upd = [0]*len(params)
    
    t0 = time.clock()
    for i in range(X.shape[0]):
        loss,grad = train(X[i],T[i],*params)
        
        for j in range(len(params)):
            params[j]-= upd[j]
            
        for j in range(len(params)):
            upd[j] = learning_rate*grad[j] + momentum*upd[j]
            
        err.append(loss)
        
    print("Epoch: %d,Loss: %.8f,Time: %.4fs" %(epoch,np.mean(err),time.clock()-t0))
    
    
# Try to predict something
x=np.random.binomial(1,0.5,n_inputs)
print("XOR Prediction")
print(x)
print(predict(x,*params))
    

