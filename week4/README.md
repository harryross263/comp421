This week we look at more of the lore of feed-forward neural nets (FFNN) - from vanilla back-prop to 'deep nets'.


# ReLUs versus sigmoids
`(Tuesday, Marcus - I will be brief)`
* the vanishing gradients issue
* ReLUs, and their gradients

# Convnets
`(Tuesday, Alex)`
* Invariance and weight tying
* What is a convolution?
* Implementation details: Padding and strides

### Resources:

* [Stanfords course on conv nets](http://cs231n.stanford.edu/)
* [Conv net visualisation](http://scs.ryerson.ca/~aharley/vis/conv/)
* [Nice explanation and pretty pics](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/)


# Varieties of SGD (stochastic gradient descent) 
`(Tuesday, lead by who?)`

Some useful links
* [Gradient descent update rules](http://sebastianruder.com/optimizing-gradient-descent/)


# Weight decay
`(Wednesday, lead by who?)`

* what it is, in the learning algorithm
* perhaps: example of it working (d.i.y. on an existing notebook? Or get picture from Bishop - see Marcus on Tuesday)
* what cost it minimizes, why this amounts to regularisation / complexity control (small weights = smooth functions)
* that's for the L2 norm of W (the sum of weights squared) - what about using the L1 norm instead?
* sparsity

# Dropout
`(Wednesday, lead by who?)`

* what it is; the intuition
* example of it working (see the first paper on it - Hinton..)
* in what sense it amounts to a kind of "regularisation"


# Review and Discussion of a research paper
`(Friday: who and who?)`

* [Alex net](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

 > ImageNet Classification with Deep Convolutional Neural Networks
 > Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
 > Published 2012 in NIPS, and already cited 6000 times
