import random

class Perceptron(object):
    ''' Defines the Perceptron object, which is comprised of some training data, current weights, an
     activation function, and methods for train()ing the Perceptron and producing a result().'''

    def __init__(self, training, activationFunction):
        '''Constructor for the Perceptron. Takes some training data represented as a list of lists,
        and a reference to an activation function'''

        # Adds the bias to the training data. self.training is equivalently a matrix
        # with column 0 set to 1 and column n-1 set to the expected output
        self.training = [[1] + t for t in training]

        # Sets the activation function
        self.activationFunction = activationFunction

        # Initialises each of the weights (including the bias) to a random number in the range [0, 1].
        # The length of the weight vector is derived from the length of the first training example
        # (FYI, see https://docs.python.org/2/tutorial/datastructures.html#list-comprehensions)
        self.weights = [random.random() for i in range(len(self.training[0])-1)]

    def train(self, iterations=10, learningRate=0.2):
        '''Trains the Perceptron. Optionally takes the number of training iterations, and a learning rate.'''

        # Repeat 'iterations' times, where iterations is 10 by default
        for iter in range(iterations):
            # For each example in the training set: compute the weighted sum, apply the activation function to that
            # weighted sum, and update the weights according to the learning rule (i.e. online learning)
            for example in self.training:
                # Lambda expression computes the weighted sum of the training set
                weightedSum = reduce(lambda x, y: x+y, [example[i]*self.weights[i] for i in range(len(self.weights))])
                activation = self.activationFunction(weightedSum)
                delta = learningRate*(example[-1] - activation)
                for i in range(len(self.weights)):
                    self.weights[i] += delta*example[i]

    def result(self):
        '''Creates a vector of outputs corresponding to each training example as a demonstration that the Perceptron
        does or does not 'fit' the training data'''

        result = []
        for example in self.training:
            weightedSum = reduce(lambda x, y: x+y, [example[i]*self.weights[i] for i in range(len(self.weights))])
            activation = self.activationFunction(weightedSum)
            result += [activation]
        return result

#############################################################
# The following code defines a suitable activation function,#
# and constructs, trains, and tests Perceptrons for each of #
# the AND, OR, NAND, and (unsuccessfully, obviously) XOR    #
# functions                                                 #
#############################################################


def stepFunction(weightedSum):
    '''Implements a simple step function such that:
        y = 1 for x > 0,
        y = 0 for x <= 0
    '''
    if weightedSum > 0:
        weightedSum = 1
    else:
        weightedSum = 0
    return weightedSum


# Tests the AND rule
AND = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
ANDnn = Perceptron(AND, stepFunction)
ANDnn.train()
assert (ANDnn.result() == [example[-1] for example in AND])  # example[-1] is the expected output

# Tests the OR rule
OR = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
ORnn = Perceptron(OR, stepFunction)
ORnn.train()
assert (ORnn.result() == [example[-1] for example in OR])

# Tests the NAND rule
NAND = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
NANDnn = Perceptron(NAND, stepFunction)
NANDnn.train()
assert (NANDnn.result() == [example[-1] for example in NAND])

# Tests the XOR case (and fails, as expected)
XOR = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
XORnn = Perceptron(XOR, stepFunction)
XORnn.train()
assert (XORnn.result() != [example[-1] for example in XOR])
