"""
Dropout: https://github.com/mdenil/dropout

"""


from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
from theano import config
from numpy.random import RandomState
floatX = config.floatX

class DropoutLayer(object):
	""" Dropout layer """
	def __init__(self, input, rng=RandomState(1234), p=0.5):
	    """
	    p is the probablity of dropping a unit
	    """

	    srng = RandomStreams(rng.randint(999999))

	    # p=1-p because 1's indicate keep and p is prob of dropping
	    mask = srng.binomial(n=1, p=1-p, size=input.shape)

	    # The cast is important because
	    # int * float32 = float64 which pulls things off the gpu
	    self.output =  input * T.cast(mask, floatX)