This lecture looks at a couple of important distributions used in machine learning.

HOW DO I COMMENT OUT STUFF LIKE THIS?!

<To read: 
     * [[http://vuw.eblib.com/patron/FullRecord.aspx?p=1591570][Textbook]]: sections ?>

# The Gaussian distribution

 * 1-D: mean and variance (1st two moments of distribution)
 * 2-D: mean is 2 numbers. Cov is 2x2 matrix. Examples, e.g. If Cov is/isn't diagonal.
 * n-D: Cov is nxn matrix: we need to invert. Cost is n^3.
 * how do we estimate mean and covariance from a data set $ X $ ?
 * mean and covariance are "sufficient statistics" for the Gaussian distribution

 * [[https://github.com/garibaldu/comp421/blob/master/notebooks/Gaussian.ipynb][ipython notebook]] showing some of these aspects.
 * Also [[http://www.inf.ed.ac.uk/teaching/courses/inf2b/lectureSchedule.html][some lecture notes]] that I (Ed) found useful. Chapter 8 is on Gaussians.





# Categorical: the "one hot" distribution

 * what's a Bernoulli?
 * so what's a Categorical? - point on simplex (what's that?)
 * [Aside: the Bernoulli is to the Binomial as Categorical is to Multinomial : second is distribution over _counts taken from the first_ ]
 * the softmax parameterisation of the categorical: %$ p_i = \frac{e^{\phi_i}}{\sum_k e^{\phi_k}} $%
 * softmax avoids restricting the parameters to [0,1] and having to "stay on the simplex"
 * so we can construct a "softmax layer" whose output is a Categorical distribution. It will be handy later...
 * notice: invariant to adding a constant to all the %$ \phi $% (so long as it's the same constant).
 * What about the 2-class case? --> show the relationship with the sigmoid function.

COMMENT OUT...
only if there's time
* MF: the mechanics of PCA
* MF: Dirichlet is a generator of multinomials
* MF: the KL divergence between two distributions -->
 
 
 
 
