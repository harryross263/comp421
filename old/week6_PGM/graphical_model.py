"""
Inference in graphical models by message passing.
Marcus Frean and Tony Vignaux
"""
import sys,copy
import numpy as np

def listLessOne(alist,k):
    return alist[0:k] + alist[k+1:len(alist)]

VERBOSE = 0  # 0 for less, 1 for more


class DiscreteGraphicalModel:

    def __init__(self,filename):

        self.variable_nodes, self.factor_nodes =  self.read_factorgraph_file(filename)
        self.all_nodes = copy.copy(self.variable_nodes)
        self.all_nodes.update(self.factor_nodes)

    def describe(self):
        print ('---------------------------------------------------------')
        print ('Descriptions of the variables:')
        print ('---------------------------------------------------------')
        for (name,node) in self.variable_nodes.items():
            node.display()
        if VERBOSE > 0:
            print ('---------------------------------------------------------')
            print ('Descriptions of the factors:')
            print ('---------------------------------------------------------')
            for (name,node) in self.factor_nodes.items():
                node.display()

            
    def read_factorgraph_file(self,filename):
        # read in a network, make the nodes, and connect them up.
        # Returns two dictionaries: names to variable nodes, and names to factor nodes
        CHECK_NORMALISATION = False

        # First we locate the list of variable nodes, and make those.
        the_variable_nodes = {}
        with open(filename,'r') as f:
            for line in f:
                if line.startswith('Variables:'):
                    vs = line.lstrip('Variables:').split(',')
                    for v in vs:
                        [name,num_states] = v.lstrip(' ').split(' ')
                        the_variable_nodes[name] = VariableNode(name,int(num_states))
        # Now start again, building up each factor node.
        the_factor_nodes = {}
        with open(filename,'r') as f:
            for line in f:
                line = line.strip('\n ') # strips newlines, and spaces, from eol.
                if (line.isspace() or (len(line)==0) or line.startswith('#')):
                    continue # ie. ignore these lines

                if line.startswith('normalised'): CHECK_NORMALISATION = True
                elif line.startswith('unnormalised'): CHECK_NORMALISATION = False

                if line.startswith('Factor:'):
                    # get its name (from the ':' onwards on this line)
                    name = line.split(':')[-1].strip()

                if line.startswith('connects to:'):
                    # get a list of its 'edges' (ie. the names of variable
                    # nodes it is connected to)
                    connected_var_names = (line.split(':')[-1]).split(',')
                    connected_variables = []
                    for v in connected_var_names: 
                        if v.strip() not in the_variable_nodes.keys():
                            sys.exit('oooooops, variable \"%s\" is not in names.' %(v))
                        connected_variables.append(the_variable_nodes[v.strip()])
                    matrix_shape = []
                    # make a matrix of the right size and fill it with negatives
                    for V in connected_variables:
                        matrix_shape.append(V.num_states)
                    factor_vals = -1.0*np.ones(tuple(matrix_shape),float) 
                    #print ('factor %s has matrix shape %s ' %(name,factor_vals.shape))
                    
                if line[0].isdigit():
                    strnums = line.rstrip().split(' ')
                    indices = map(int, strnums[:-1]) #everything except the last entry
                    val = float(strnums[-1])
                    factor_vals[tuple(indices)] = val
            
                    if np.min(np.ravel(factor_vals)) >= 0.0:
                        # this only happens when we have updated ALL the values,
                        # and so are ready to make a new FactorNode.
                        if CHECK_NORMALISATION:
                            # ensure the factor entries are normalised
                            # w.r.t. the final index.
                            fT = factor_vals.transpose()
                            if np.alltrue(np.equal(factor_vals, (fT / fT.sum(0)).transpose())):
                                print ('Factor %s already normalised' %(name))
                            else:
                                print ('%s was not normalised in %s: IS BEING NORMALISED...' %(name,filename))
                                factor_vals = (fT / fT.sum(0)).transpose()

                        the_factor_nodes[name] = FactorNode(name,
                                                            connected_variables,
                                                            factor_vals)

        return the_variable_nodes, the_factor_nodes


    def propagate_messages(self):
        # Initialise all messages: every terminal node dings its neighbour.
        print ('Initialising all messages')
        for (name,node) in self.all_nodes.items():
            if len(node.edges) == 1: # ie. it has one edge so it's a terminal node.
                node.initialDing()
        print ('Messages propagated.\n\n')


    def observe(self,variable_name,state):
        # If the named variable is observed to be in the given state, 
        # 1. It must ding all its neighbours with obs.
        # 2. It must somehow 'disable' incoming dings so they have no effect.
        # 3. The variable has to 'know' its own observed value.
        # ALL THESE will occur if this variable node simply acquires a new
        # (leaf) FactorNode, consisting of zeros-bar-one (which dings back self
        # once). So basically an observation IS a mostly-zeros factor!
        print ('OBSERVATION: %s is seen to be in state %d' %(variable_name, state))
        variable = self.all_nodes[variable_name]
        name = 'factor representing obs of %s' %(variable.name)
        factor_vals = np.zeros((variable.num_states),float)
        factor_vals[state] = 1.0
        obs = ObservationNode(name,variable,factor_vals)
        # include this weird factor in the overall lists of nodes.
        self.factor_nodes[obs.name] = obs
        self.all_nodes[obs.name] = obs




class Node:
    def __init__(self,name=''):
        self.name=name
            
    def __str__(self):
        return self.name

    def ding(self, toNode, outboundMsg, dingChain):
        # This simply alerts a neighbour (toNode) that it needs to
        # respond to a message (outboundMsg)
        if VERBOSE > 0:
            print ('\t node %s dings %s (%d) with msg %s' % (self.name, toNode.name, dingChain, outboundMsg))
        if dingChain < 10:
            dingChain = dingChain + 1
            r = toNode.respondToMessage(self, outboundMsg, dingChain)
        
    def respondToMessage(self,fromNode,inboundMsg, dingChain):
        # A node responds to an inbound message from fromNode, by
        # going through its other neighbours, recalculating messages
        # to them, and letting them know (via ding).
        k=self.edges.index(fromNode) # looks up the index
                                     # corresponding to fromNode
        self.msg[k]=inboundMsg
        receivers = listLessOne(self.edges,k)
        for r in receivers:
            i=self.edges.index(r)
            newMsg=self.calcMessage(i)
            self.ding(r,newMsg,dingChain)

        
class VariableNode(Node):
    def __init__(self,name='',num_states=1):
        Node.__init__(self,name)
        self.num_states=num_states
        self.edges=[]
        self.msg=[1]*len(self.edges)
        self.observed=False
            
    def __str__(self):
        return self.name

    def display(self):
        print ('node %s' % (self.name))
        if self.observed:
            print ('\t Has been OBSERVED')
        else:
            for i in range(len(self.edges)):
                print ('\t msg from factor %s is %s' %(self.edges[i], str(self.msg[i])))
        p = np.product(np.asarray(self.msg),0)
        print ('\t posterior: %s' % (str(p/np.sum(p))))
        print ('----------')


    def calcMessage(self,i):
        newMsg=np.product(np.asarray(listLessOne(self.msg,i)),0)
        return newMsg

    def initialDing(self):
        # VariableNodes that are terminal nodes need to ding their 
        # edge with a message consisting of all ones.
        print ('\t Initial ding from terminal node %s to %s' % (self.name, self.edges[0].name))
        dingChain = 0
        self.ding(self.edges[0], np.ones((self.num_states),float), dingChain)


class FactorNode(Node):
    def __init__(self,name='',edges=[],phi=[]):
        Node.__init__(self,name)
        self.edges=edges

        self.msg=[1]*len(self.edges)
        for i in range(len(self.edges)):
            neighbour=self.edges[i]
            self.msg[i] = np.ravel(np.ones((1,neighbour.num_states),float)) # vector of ones
            
            # update the neighbour's edges and msgs
            neighbour.edges.append(self)
            neighbour.msg.append(self.msg[i])
        self.phi=phi
        # Check that dimensions of phi match the num_states's of variables.
        for i in range(len(edges)):
            if not((self.edges[i]).num_states == self.phi.shape[i]):
                print ('Ooops: shape of',self.name,'phi doesnt match its variables.')
                print ('Shape[',i,'] is',self.phi.shape[i])
                print ('num_states of edge',self.edges[i],'is',(self.edges[i]).num_states)
                # ........AND WE SHOULD QUIT HERE, WITH ERROR MESSAGE...
                stderr.write('There is a mismatch between size of phi and of a message')
                

    def calcMessage(self,i):
        # We need to be able to leave out one dimension at will.
        # First, we rotate phi around so it's got the i-th axis first,
        # followed by the others in ascending order:
        nPhiDims = len(self.phi.shape)
        axesorder = [i] + listLessOne(list(range(nPhiDims)),i)
        z=np.transpose(self.phi, axes=(axesorder))
        index=list(range(len(self.msg)-1))
        # Go through the other messages and "integrate them out." Each
        # time a variable is summed out like this the dimensionality
        # of z (i.e. phi) goes down by one. We're summing out the
        # right-most dimension of z. We go through msg in reverse
        # order so that the length of the msg and the *rightmost*
        # dimension of z match.
        index.reverse()
        othermsg = listLessOne(self.msg,i)
        for j in index:
            y=othermsg[j]*z
            z=np.transpose(np.sum(np.transpose(y),0))
        return z

    def initialDing(self):
        # FactorNodes that are terminal nodes need to ding their (one)
        # edge with the "message" phi.
        print ('\t Initial ding from terminal node %s --> %s' % (self.name, self.edges[0].name))
        dingChain = 0
        self.ding(self.edges[0], self.phi, dingChain)

    def __str__(self):
        return self.name

    def display(self):
        print (self.name)
        for i in range(len(self.edges)):
            print ('intray %s msg: %s' % (self.edges[i],self.msg[i]))
        print ('phi is: %s' % (str(self.phi)))
        print ('-----------')

    
class ObservationNode(FactorNode):
    # Called if a variable is observed - obs is the resulting vector.
    def __init__(self,name,observedNode,obs=[]):
        FactorNode.__init__(self,name,[observedNode],obs)
        self.edges=[observedNode]
        self.initialDing()
        observedNode.observed = True # just for humans, not used algorithmically


if __name__ == '__main__':
    print ('just a test................')
    infile = 'simple_network.txt'
    net = DiscreteGraphicalModel(infile)
    net.propagate_messages()
    net.describe()
    net.observe('rain',1)
    net.describe()


