
import numpy as np    
      
def sigmoid(X):
    return 1./(1.+np.exp(-X))    
'''
derivative of sigmoid function
'''
def dSigmoid(X):
    return sigmoid(X)*(1-sigmoid(X))



def tanh(X):
    return np.tanh(X)

def dTanh(X):
    return 1- np.tanh(X)**2

'''
Rectified Linear Unit
'''

def reLU(X):
    return np.log(1+np.exp(X))

def dReLU(X):
    return sigmoid(X)