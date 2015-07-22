import numpy as np
from ActivationFunctions import * 
import matplotlib.pyplot as plt

class MultiLayerPerceptron():
    '''
    Multi Layer Perceptron
    '''
    
    def __init__(self ,netStruct ,actv=None ,dActv=None ,eta=0.2):
        
        '''        
        parameters
              netStruct: sequence
                    dimensionality of the network weights. number of nodes in the first(input) layer equals to the number of explanatory variables, e.g. netStruct=(3,2,1)  is a network with 3 input nodes, one hidden layer with  2 hidden units, and one output node
              actv: function
                    activation function
              dActv: function
                    first derivative of the activation function w.r.t its input
              eta: scalar
                    learning rate used in back propagation                
        '''
    
        self.eta = eta
        #activation function
        self.actv = actv
        # derivative of the activation function
        self.dActc = dActv
        # list of network layers
        self.weights = []
        #initialize weights in each layer i according to the specified network structure                  
        for i in xrange( len( netStruct )-1 ):
            # calculating the number of weights in each layer. an extra node is added to each layer to represent the bias 
            layerShape,layerSize= ( netStruct[i]+1 , netStruct[i+1] ), ( netStruct[i]+1 ) * ( netStruct[i+1] )
            #weights are initialized randomly 
            self.weights.append( np.random.uniform( size=layerSize ).reshape( layerShape ) )
            #self.weights.append( np.ones(shape= layerShape ) )


    def predict(self,X):       
        '''
        parameters
           X: 2d numpy array , shape: (numberofFeatureVectors,numberofFeatures)
        returns
           the network response for a given set of feature vectors          
        ''' 
        
        return self.forwardPass(X)[-1]



        
    def forwardPass(self,X):   
        '''                    
        parameters
            X: 2d numpy array, shape: (1,numberofFeatures ) 
            feature vector
        returns
            outputs of network layers for the given data point     
        '''   
        X = np.atleast_2d(X)  
        layersOut=[]
        # an extra node with it's value set to 1 represents the bias 
        currentlayerOut=np.vstack( (X.T,np.ones( X.shape[0] ) ) ).T
        # output of the first layer is the feature vector(s) 
        layersOut.append( currentlayerOut )
        #starting from layer 0 
        i=0
        while i < len( self.weights )-1:
            #pass the output of each layer forward 
            currentlayerOut =self.actv( currentlayerOut.dot( self.weights[i] ) )
            # add bias 
            currentlayerOut = np.vstack( ( currentlayerOut.T , np.ones( currentlayerOut.shape[0] ) ) ).T
            
            layersOut.append( currentlayerOut )
            i+=1            
        currentlayerOut= self.actv( currentlayerOut.dot( self.weights[-1] ) ) 
        layersOut.append( currentlayerOut )            
        return layersOut



    
    def train(self ,xTrain ,yTrain ,epocs=1000,plotError=True ):
        '''
        parameters
            xTrain:2d numpy array, shape: (numberofFeatureVectors, numberofFeatures)
            yTrain:2d numpy array, shape: (numberofFeatureVectors, numberofOutputs)
        trains the network using stochastic gradient descent
        '''
        numSamples = xTrain.shape[0]
        
        indexList = [i for i in range( numSamples )]
        errorList=[]
        for epoc in xrange(epocs):    
            #shuffle the training set                    
            np.random.shuffle(indexList )
            # save the current weights  
            #oldWeights = [ item for item in self.weights ]            
            for index in indexList:
                errors=[]
                layersOut = self.forwardPass( xTrain[index] )
                error =  layersOut[-1] - yTrain[index]                 
                self.backwardPass( error ,layersOut[:-1] )
                if plotError:   
                    errors.append(np.abs( np.sum( error ) ) )
            if plotError:
                errorList.append(np.average( np.asarray(errors) ) )
        if plotError:            
            plt.plot(errorList,)
            plt.title('Training Error')
            plt.ylabel('Error')
            plt.xlabel('Iteration')
            plt.show()
                
    def backwardPass(self ,error ,layersOut ):
        '''
        updates the network weights using backpropagation 
                     
        '''
        #we  already have the output error. So output layer is not considered 
        i=len( layersOut )-1
        #go through all layers in reverse order
        while i>=0:
            # calculate the deltas 
            dOut_dIn = self.dActc( layersOut[i].dot( self.weights[i] ) )*error
            #  dE_dW = dOut_dIn* dIn_dW
            dE_dw = layersOut[i].T.dot( dOut_dIn )   
            #propagate the errors backward to the output of the previous layer nodes, except for the bias node    
            error = error.dot(self.weights[i].T)[:,:-1]
            # update the weights in the layer i        
            self.weights[i] -= self.eta* dE_dw
            i-=1
             


'''
#test
#binary class problem xtrain[0] < xtrain[1] -> class 0  otherwise class 1

    
model=MultiLayerPerceptron(actv=sigmoid,dActv=dSigmoid,netStruct=(2,5,1),eta=0.2)


xtrain = np.asarray([[1.3,0],[1.1,0.5],[3,1.7],[4,2],[2,2.5],[4,7.8],[9,9.1]])

ytrain =np.asarray([[1],[1],[1],[1],[0],[0],[0]])

model.train(xtrain,ytrain,epocs=1000)
print  model.predict(xtrain)
'''
            