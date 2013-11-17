"""
Dropout: https://github.com/mdenil/dropout

"""


from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
from theano import config
from numpy.random import RandomState
floatX = config.floatX


def dropout(layer_output, rng=RandomState(1234), p=0.5):
    """
    p is the probablity of dropping a unit
    """

    srng = RandomStreams(rng.randint(999999))

    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer_output.shape)

    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    return layer_output * T.cast(mask, floatX)