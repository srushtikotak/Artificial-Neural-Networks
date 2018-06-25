import numpy as np

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

class ANN(object):
    def __init__(self, epoch, lr, input, hidden, output):
        #Variable initialization
        self.epoch = epoch #Setting training iterations
        self.lr = lr #Setting learning rate
        self.input_dim = input #number of features in data set
        self.hidden_dim = hidden #number of hidden layers neurons
        self.output_dim = output #number of neurons at output layer

    def forward_backward(self, X, y):
        #weight and bias initialization
        wh = np.random.uniform(size=(self.input_dim,self.hidden_dim))
        bh = np.random.uniform(size=(1,self.hidden_dim))
        wout = np.random.uniform(size=(self.hidden_dim,self.output_dim))
        bout = np.random.uniform(size=(1,self.output_dim))

        for i in range(self.epoch):
            #Forward Propogation
            hidden_act_1 = np.dot(X,wh)
            hidden_act = hidden_act_1 + bh
            hiddenlayer_activations = sigmoid(hidden_act)
            output_act_1 = np.dot(hiddenlayer_activations,wout)
            output_act = output_act_1+ bout
            output = sigmoid(output_act)
            #print output.shape
            
            #Backpropagation
            E = y - output
            slope_output_layer = derivatives_sigmoid(output)
            slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
            d_output = E * slope_output_layer
            Error_at_hidden_layer = d_output.dot(wout.T)
            d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
            wout += hiddenlayer_activations.T.dot(d_output) *self.lr
            bout += np.sum(d_output, axis=0,keepdims=True) *self.lr
            wh += X.T.dot(d_hiddenlayer) *self.lr
            bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *self.lr
        
        print 'predicted \n', output
        print 'actual \n', y

