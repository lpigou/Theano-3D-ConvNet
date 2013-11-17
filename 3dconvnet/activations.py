"""
Activation functions

"""


import theano


tanh = theano.tensor.tanh
sigmoid = theano.tensor.nnet.sigmoid


def recifier(self, X):
    """Rectified linear units"""
    return X * (X > 0.) 