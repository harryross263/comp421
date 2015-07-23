
import numpy as np    
      
def sigmoid(X):
    return 1./(1.+np.exp(-X))    
'''
derivative of sigmoid function
'''
def dSigmoid(X):
    return sigmoid(X)*(1-sigmoid(X))

