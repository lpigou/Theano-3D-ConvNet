"""
Activation functions

"""


import theano


tanh = theano.tensor.tanh
sigmoid = theano.tensor.nnet.sigmoid

@staticmethod
def rectifier(X):
    """Rectified linear units"""
    return X * (X > 0.) 