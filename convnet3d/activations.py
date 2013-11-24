"""
Activation functions

"""


import theano.tensor as T


tanh = T.tanh
sigmoid = T.nnet.sigmoid
softplus = T.nnet.softplus

# @staticmethod
def relu(X):
    """Rectified linear units (relu)"""
    return T.maximum(0,X)