# First lecture
This lecture looks at a couple of important distributions used in machine learning.

## The Gaussian distribution

 * 1-D: mean and variance (1st two moments of distribution)
 * 2-D: mean is 2 numbers. Cov is 2x2 matrix. Examples, e.g. If Cov is/isn't diagonal.
 * n-D: Cov is nxn matrix: we need to invert. Cost is $n^3$.
 * how do we estimate mean and covariance from a data set $ X $ ?
 * mean and covariance are "sufficient statistics" for the Gaussian distribution

 * [ipython notebook](https://github.com/garibaldu/comp421/blob/master/notebooks/Gaussian.ipynb) showing some of these aspects.
 * Also [http://www.inf.ed.ac.uk/teaching/courses/inf2b/lectureSchedule.html](some lecture notes that I (Ed) found useful. Chapter 8 is on Gaussians.

## Categorical: the "one hot" distribution

 * what's a Bernoulli?
 * so what's a Categorical? - point on simplex (what's that?)
 * [Aside: the Bernoulli is to the Binomial as Categorical is to Multinomial : second is distribution over _counts taken from the first_ ]
 * the softmax parameterisation of the categorical: $ p_i = \frac{e^{\phi_i}}{\sum_k e^{\phi_k}} $
 * softmax avoids restricting the parameters to [0,1] and having to "stay on the simplex"
 * so we can construct a "softmax layer" whose output is a Categorical distribution. It will be handy later...
 * notice: invariant to adding a constant to all the %$ \phi $% (so long as it's the same constant).
 * What about the 2-class case? show the relationship with the sigmoid function.

< only if there is time * MF: the mechanics of PCA * MF: Dirichlet is a generator of multinomials * MF: the KL divergence between two distributions >
 
***
# Second lecture
< presented by Gerard and Catherine >

This lecture looks at a couple of the common _cost functions_ or _loss functions_ used in machine learning: these are functions which the process of _learning_ seeks to _optimize_ (either minimize or maximize).

Sometimes the choice of what to optimize can seem rather arbitrary. A nice way to be more principled is to explicitly consider a _probabilistic model_ and then ask the learner to maximize the _log likelihood_, which is the (log of the) probability that, given the input patterns one after another, the learner would produce the same outputs as are in the training set. Here we look at a couple of examples.



# Intro: (MF)
 * X --> T in the training set, but X --> Y (prediction) under a model having parameters w. We want Y to match T...
 * One way to view this: think of Y as specifying a _probability distribution_ and ask "What is the likelihood of generating T exactly, under this distribution?"
  * The Aim of learning could be to maximize that likelihood.
  * under the "i.i.d. assumption" the likelihood is a _product_ over training items so the log likelihood of the training data is a sum over training items


# A cost function for regression tasks: the sum of squared errors

   * what that is
   * minimizing it in the case of a linear model, using gradient descent ---> "the delta rule" 
   * in what sense does it make sense? log of likelihood under a Gaussian noise model....
   * EX: use the i.i.d. assumption and a Gaussian model to show the log likelihood == sum of squared errors ! 

(we leave the topic of regularisation - penalising "complexity", e.g. "weight decay" - to the next week).


# A cost function for classification tasks: "Cross-entropy"
 * likelihood of getting data correct, under a model generating a Categorical distribution
 * for just 2 classes: log likelihood under the categorical distribution --> "cross entropy".
 
< Possible exercise: What is the learning rule that results for sigmoids (and softmax more generally) >
 
***
# Third lecture
no such thing - Marcus was away so no lecture.
