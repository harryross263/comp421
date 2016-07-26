# Week 1 - introductions and straw men

First lecture was an introduction to what we're going to cover, and how we could assess it.


< presented by Harry and Luke >

Some "straw man" machine learning algorithms (ie. very simple, but not very good in general).

Talking points:

# kNN
 * the simplest classifier you can imagine
 * Nearest neighbours, e.g. k=1
 * no "learning" at all, but could pre-cook the stored data in order to make access faster
 * what's the complexity?
 * pros?
    * simple to write
    * can never be _too_ wrong!
 * cons?
    * arbitrariness of the distance metric we use
    * need to keep ALL the data around in memory
    * new item seems to need to be compared to all the previous data (but may be able to reduce somewhat with smarts)
  
# Perceptron
 * another very simple classifier
 * postponed until the next lecture - short of time


# k-means
 * a clustering algorithm
 * we ran it on the blackboards...
 * something sensible is being minimized - what is it?
 * algorithm terminates when boundaries stop moving
 * local optima are possible, and a threat
 * reduces data to a "cartoon" form of 1-of-k index

# PCA
 * an algorithm for "dimensionality reduction"
 * 2-d cloud example
 * 3-d chair example (!)
 * finds the k directions that capture most of the variability in the data
 * you can then project the original data onto those directions, thus "compressing" it
 * fine if your data is actually "blobby", but could be very misleading if it is (say) boomerang shaped, or clumpy
 * [[https://github.com/garibaldu/comp421/blob/master/notebooks/Simple_PCA_example.ipynb][ipython notebook]] showing some basic PCA in action.
Noted: both k-means and PCA make "cartoon" / reduced forms of the original data.

# TODO: 
 * note kNN and k-means are _non-parametric_, whereas perceptron and PCA are _parametric_
 * _using_ a non-parametric method may be costly, whereas _learning_ a parametric method may be costly.

---

< presented by Marcus >

This lecture picks up our last "straw man", namely the humble Perceptron.

To read: 
 * [Textbook](http://vuw.eblib.com/patron/FullRecord.aspx?p=1591570][Textbook) : sections 2.1 and 2.2
 * [Textbook](http://vuw.eblib.com/patron/FullRecord.aspx?p=1591570][Textbook): all of chapter 3, except for 3.4.1. (The later parts of the chapter are useful if you're unfamiliar with python - there is also a section in an appendix).


 * data is a big matrix of inputs X mapping (in "supervised learning" at least) to a matrix of "targets" Y
 * we try to learn a mapping that takes X to Y
 * this is done in the expectation that a good mapping will yield a sensible result y when given a novel x.

# Perceptrons

 * the most basic of parameterized models for the mapping X --> Y
 * weighted sum followed by thresholding operation
 * bias weight is good, acts as learnable threshold, can be treated as extra "zeroth" input that is always 1
 * perceptron's decision surface (where output changes) is a hyperplane, perpendicular to the weights vector
 * "learning" is movement of that hyperplane, in effect
 * Perceptron Learning Rule (often called the "delta rule") - guaranteed to find a separating hyperplane, if one exists
 * There's an [ipython notebook](https://github.com/garibaldu/comp421/blob/master/notebooks/super-simple-Perceptron.ipynb) showing some of these aspects - download and run the last cell several times (from its random initial weight values) to see the effect of learning the weights on the hyperplane dividing the space.
 * XOR: a simple problem where separating hyperplane doesn't exist

# The Curse of Dimensionality

In high dimensional spaces:
 * relative volume of hypersphere vs hypercube is miniscule: almost all the volume is "in the corners"
 * random vectors are orthogonal
 * typical distance between any two vectors tends to be the same, as a result
 * we will come across numerous consequences for machine learning, e.g. kNN says nearest point is privileged, but all points are becoming equally "near"!

# Comment

We've met 4 straw-man machine learning algorithms lately:

| _                | *supervised* | *unsupervised* |
| *non-parametric* | kNN            | k means |
| *parametric*     | perceptron     | PCA     |
 
(look up parametric versus non-parametric).
