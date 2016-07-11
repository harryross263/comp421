
# coding: utf-8

# # horsing around with the backprop algorithm
# Marcus started this see how quickly he could get backprop to stand up.
# 
# Note the use of "checkgrad", which exhaustively confirms that the gradient calculation is in fact correct - not something to run all the time but a useful check to have.
# 
# Issues:
#   * the neural net has no biases yet
#   * the learning problem is just random - better if we could read in a training set

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rng
np.set_printoptions(precision = 2, suppress = True)


# ### specify a neuron transfer function ('funk'), and its derivative

# In[2]:

# THESE FUNKERS MUST MATCH ONE ANOTHER................

def funk( phi ):  
    # phi is always going to be a weighted sum (probably a matrix of).
    x = 1.0/ (1.0 + np.exp(-phi))
    
    #ALT: rectified linear goes like this
    #x = phi * (phi>0.0)
    return x

def dfunk_from_funk( x ):  # MUST MATCH WHAT YOU PUT HERE with the funk function.
    # This is the gradient of the transfer function (funk)
    # with respect to "phi", the weighted sum of inputs to the neuron.
    # But the input argument isn't phi here - it's the function value itself.

    dx = x*(1-x)
    #ALT: rectified linear goes like this
    #dx = 1.0*(x>0.0)
    return dx


# ### get or make some training data
# Got to have something to work on.

# In[3]:

# I'm going to be dumb here and make them from my very own random perceptrons!
# However you do it, call the input patterns "inpats" (each row is a pattern), and the output patterns "targets".
Nins, Nouts, Npats = 8, 2, 100
inpats = rng.normal(0,1,size=(Npats,Nins))
tmpNhids = 10
tmp_weights = rng.normal(0,1,size=(Nins,tmpNhids))
hidphi = np.dot(inpats, tmp_weights)
tmp_weights = rng.normal(0,1,size=(tmpNhids,Nouts))
phi = np.dot(funk(hidphi), tmp_weights)
targets = 1.0* (funk(phi) > 0.5)
print (inpats[:3, :])
print (targets[:3, :])


# ### The function we're climbing

# In[4]:

def calc_goodness(outputs, targets):
    # outputs is a matrix (Npats, Nouts), and targets is what we'd like those to be.
    error = targets - outputs
    Good_vec = -0.5*np.power(error,2.0) # inverted parabola centered on the target outputs
    dGood_vec = error # e.g. if output is too low, this should be positive.
    return Good_vec.sum(), Good_vec, dGood_vec


# ### set the network's architecture

# In[5]:

Npats = inpats.shape[0]
architecture = [inpats.shape[1], 5, 3, targets.shape[1]]
#architecture = [inpats.shape[1], targets.shape[1]]
print ('There are this many neurons in each layer: ', architecture)  # a list of the number of neurons in each layer


# In[6]:

X = [inpats] 
# X is going to be a list giving the activations of successive layers. 
# Each one is a matrix, whose columns are the neurons in that layer.
# Each row in the matrix corresponds to a training item.
# So all the matrices in the list X will have the same number of rows.

for L in range(1, len(architecture)):
    X.append(np.zeros(shape=(Npats, architecture[L]), dtype=float))

for L in range(len(architecture)): 
    print('layer %d activations have shape ' %(L), X[L].shape)


# ### set up the weights

# In[7]:

# Then we have the weights. I'm going to index weight layer by the layer they're *going* towards.
# So I'll have a zeroth weight layer for sanity, but it's going to be empty!
# NOTE: no implementation of bias weights here, yet!
W  = [np.array(None)]
dW = [np.array(None)]
for L in range(1,len(X)):
    init_weights_scale = 0.1  #1/np.sqrt((X[L].shape()).max())

    Nins = X[L-1].shape[1]
    Nouts = X[L].shape[1]
    W.append(init_weights_scale * rng.normal(0,1,size=(Nouts, Nins)) )
    dW.append(0.0 * np.copy(W[L]))

for L in range(len(W)):
    print('layer %d weights have shape ' %(L), W[L].shape)


# ### forward pass

# In[8]:

def forward_pass(X, W):
    for L in range(1,len(X)):
        x = X[L-1].transpose()
        # print (L, W[L].shape, x.shape)
        X[L] = funk(np.dot(W[L], x).transpose())
    return X


# ### backward pass

# In[9]:

def backward_pass(X, W, dW, targets):
    good_sum, good_vec, dgood = calc_goodness(X[-1], targets)
    epsilon = dgood
    npats = X[0].shape[0]
    for L in range(len(X)-1,0,-1):
        psi = epsilon * dfunk_from_funk(X[L]) # elt-wise multiply
        n1, n2 = X[L-1].shape[1], psi.shape[1]
        A = np.tile(X[L-1],n2).reshape(npats,n2,n1)
        B = np.repeat(psi,n1).reshape(npats,n2,n1)        
        dW[L] = (A*B).sum(0) # outer product multiply
        epsilon = np.dot(psi, W[L]) # inner product multiply
    return dW


# In[10]:

X = forward_pass(X, W)
dW = backward_pass(X, W, dW, targets)


# In[11]:

def checkgrad(dW, X, W, targets):
    # Check the gradient directly, via perturbations to every weight.
    # This is completely daft in practical terms, but very useful for debugging.
    # ie. it tells you whether your backprop of errors really is returning the true gradient.
    tiny = 0.0001
    
    dW_test = [np.array(None)]
    for L in range(1,len(W)):
        dW_test.append(0.0*np.copy(W[L]))
    
    X = forward_pass(X,W)
    base_good, tmp1, tmp2 = calc_goodness(X[-1], targets)
    
    for L in range(1,len(X)):
        for j in range(W[L].shape[0]): # index of destination node
            for i in range(W[L].shape[1]): # index of origin node
                # perturb that weight
                (W[L])[j,i] = (W[L])[j,i] + tiny
                # compute and store the empirical gradient estimate
                X = forward_pass(X,W)
                tmp_good, tmp1, tmp2 = calc_goodness(X[-1], targets)
                (dW_test[L])[j,i] = (tmp_good - base_good)/tiny                
                # unperturb the weight
                (W[L])[j,i] = (W[L])[j,i] - tiny
                
    # show the result?
    for L in range(1,len(X)):
        print ('-------------- layer %d --------------' %(L))
        print ('calculated gradients:')
        print (dW[L])
        print ('empirical gradients:')
        print (dW_test[L])


# In[12]:

checkgrad(dW, X, W, targets)


# ## yay.
# The gradient seems to be right for the full MLP, so that's... progress!
# 
# Let's try learning the problem then....

# In[13]:

def learn(X, W, dW, targets, learning_rate=0.01, momentum=0.1, num_steps=1):
    # note dW and prev_change are of the same size as W - we'll make space for them first
    times, vals = [], []
    next_time = 0
    
    prev_change = [np.array(None)]
    for L in range(1,len(X)):
        prev_change.append(0.0 * np.copy(W[L]))
    
    # now for the learning iterations
    for step in range(num_steps):
        X = forward_pass(X,W)
        
        # this is just record-keeping.......
        if step == next_time:
            good_sum, good_vec, dgood = calc_goodness(X[-1], targets)
            vals.append(good_sum)
            times.append(step)
            next_time = step + 10

        dW = backward_pass(X, W, dW, targets)
        for L in range(1,len(X)):
            change =  (learning_rate * dW[L])  +  (momentum * prev_change[L])
            W[L] = W[L] + change
            prev_change[L] = change


    return W, times, vals


# In[14]:

W, vals, times = learn(X, W, dW, targets, learning_rate=0.01, momentum=0.5, num_steps=10000)
plt.plot(vals, times)


# In[ ]:



