import sys
from graphical_model import *
    

if __name__ == '__main__':

    infile = 'simple_network.txt'

    net = DiscreteGraphicalModel(infile)
    net.propagate_messages()

    net.describe()
    net.observe('wet',1)
    net.describe()


