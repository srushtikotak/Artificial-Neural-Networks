import numpy as np
import nn

def demo():
    
    X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1],[1,0,1,0],[1,0,1,1],[0,1,0,1]])
    y=np.array([[1],[1],[0],[1],[1],[0]])

    NN = nn.ANN(5000, 0.1, X.shape[1], 3, 1)
    NN.forward_backward(X,y)
        
if __name__ == '__main__':
    demo()
