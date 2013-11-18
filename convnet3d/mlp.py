"""
MLP Layers using Theano

    LogRegr
    HiddenLayer
    DropoutLayer
"""

from numpy import zeros, sqrt
from numpy.random import RandomState
from theano import shared, config, _asarray
from theano.tensor.nnet import  softmax, sigmoid
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
floatX = config.floatX


class LogRegr(object):
    """ Logistic Regression Layer, Top layer, Softmax layer, Output layer """

    def __init__(self, input, n_in, n_out, layer_name="LogReg", W=None, b=None, 
        borrow=True):

        # Weigth matrix W
        if W == None:
            self.W = shared(zeros((n_in, n_out), dtype=floatX), 
                name=layer_name+"_W",
                borrow=borrow)
        else: 
            self.W = shared(W, name=layer_name+"_W", borrow=borrow)

        # Bias vector
        if b == None:
            self.b = shared(zeros((n_out,), dtype=floatX),
                name=layer_name+"_b",
                borrow=borrow)
        else: 
            self.b = shared(b, name=layer_name+"_b", borrow=borrow)

        # Vector of prediction probabilities
        self.p_y_given_x = softmax(T.dot(input, self.W) + self.b)
        # Prediction
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # Parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """ Cost function: negative log likelihood """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """ Errors over the total number of examples (in the minibatch) """
        return T.mean(T.neq(self.y_pred, y))


class HiddenLayer(object):
    """ Hidden Layer """

    def __init__(self, input, n_in, n_out, activation, rng=RandomState(1234), 
        layer_name="HiddenLayer", W=None, b=None, borrow=True):

        if W == None:
            # uniformly sampled W
            low = -sqrt(6. / (n_in + n_out))
            high = sqrt(6. / (n_in + n_out))
            values = rng.uniform(low=low, high=high, size=(n_in, n_out))
            W_val = _asarray(values, dtype=floatX)
            if activation == sigmoid: W_val *= 4
            self.W = shared(value=W_val, borrow=borrow, name=layer_name+'_W')
        else: 
            self.W = shared(value=W, borrow=borrow, name=layer_name+'_W')

        if b == None:
            # Initialize b with zeros
            self.b = shared(value=zeros((n_out,), dtype=floatX), borrow=True)
        else: 
            self.b = shared(b, name=layer_name+"_b", borrow=borrow)

        # Parameters of the model
        self.params = [self.W, self.b]
        # Output of the hidden layer
        self.output = activation(T.dot(input, self.W) + self.b)


class DropoutLayer(object):
    """ Dropout layer: https://github.com/mdenil/dropout """

    def __init__(self, input, rng=RandomState(1234), p=0.5):
        """
        p is the probablity of dropping a unit
        """

        srng = RandomStreams(rng.randint(999999))

        # p=1-p because 1's indicate keep and p is prob of dropping
        mask = srng.binomial(n=1, p=1-p, size=input.shape)

        # The cast is important because
        # int * float32 = float64 which pulls things off the gpu
        self.output = input * T.cast(mask, floatX)