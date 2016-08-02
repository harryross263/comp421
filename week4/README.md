This week we look at more of the lore of feed-forward neural nets (FFNN) - from vanilla back-prop to 'deep nets'.


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
`(Tuesday, lead by Davis)`

Some useful links
* [Gradient descent update rules](http://sebastianruder.com/optimizing-gradient-descent/)



# ReLUs versus sigmoids
`(Wednesday, Marcus - I will be brief)`
* the vanishing gradients issue
* ReLUs, and their gradients


# Weight decay
`(Wednesday, lead by Ben)`

* the point: over-fitting, i.e. Defending against the arbitrariness of initialisation of W.
* idea: complexity control = penalties added to the loss function = *regularisation*
* the penalties could be on the weights values, e.g. prefer "smaller" weights
* Final weights will be a compromise between getting data right and keeping weights small. (Aside: in a net with sigmoid hiddens, small weights means smooth functions).
* specifically: the 2-norm leads (via gradient-following) to "weight decay". 
* how to set the amount of decay?
* what about the 1-norm instead? (leads to sparsity: often desirable).
* Nb. possible 2nd point to this: regularisation can improve convergence (perhaps by removing local optima).



# Dropout
`(Wednesday, lead by Kelsey)`

* what it is; the intuition
* example of it working (see the first paper on it - Hinton..)
* in what sense it amounts to a kind of "regularisation"



# Review and Discussion of a research paper
`(Friday: lead by Scott and who?)  (also: Luke E and Max volunteered - second paper? t.b.d.)`

* [Alex net](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

 > ImageNet Classification with Deep Convolutional Neural Networks
 > Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
 > Published 2012 in NIPS, and already cited 6000 times
