"""
3D ConvNet layers using Theano, Pylearn and Numpy

ConvLayer: convolutions, filter bank
NormLayer: normalization (LCN, GCN, local mean subtraction)
PoolLayer: pooling, subsampling
RectLayer: rectification (absolute value)

"""


from conv3d2d import conv3d
from max_pool_3d import max_pool_3d
from dropout import dropout

from numpy import asarray, sqrt, prod, ones, floor, repeat
from numpy.random import RandomState

from pylearn2.datasets.preprocessing import gaussian_filter
from pylearn2.utils import sharedX

from theano.tensor.nnet import conv2d
from theano import shared, config
import theano.tensor as T
floatX = config.floatX


class ConvLayer(object):
    """ Convolutional layer, Filter Bank Layer """

    def __init__(self, input, n_in_maps, n_out_maps, kernel_shape, input_shape, 
        batch_size, activation, layer_name="Conv", rng=RandomState(1234), 
        borrow=True, W=None, g=None, b=None, use_bias=True, use_gain=False):

        """
        input_shape: (frames, height, width)
        kernel_shape: (frames, height, width)

        W_shape: (out, in, kern_frames, kern_height, kern_width)
        """

        self.__dict__.update(locals())
        del self.self
        
        # init W
        if W != None: W_val = W
        else: 
            # fan in: filter time x filter height x filter width x input maps
            fan_in = prod(kernel_shape)*n_in_maps
            norm_scale = 2. * sqrt( 1. / fan_in )
            W_shape = (n_out_maps, n_in_maps)+kernel_shape
            W_val = asarray(rng.normal(loc=0, scale=norm_scale, size=W_shape), \
                        dtype=floatX)
        self.W = shared(value=W_val, borrow=borrow, name=layer_name+'_W')
        self.params = [self.W]

        # init bias
        if use_bias:  
            if b != None: b_val = b
            else: b_val = ones((n_out_maps,), dtype=floatX)
            self.b = shared(b_val, name=layer_name+"_b", borrow=borrow)
            self.params.append(self.b)

        # init gain
        if use_gain: 
            if g != None: g_val = g
            else: g_val = ones((n_out_maps,), dtype=floatX)
            self.g = shared(g_val, name=layer_name+"_g", borrow=borrow)
            self.params.append(self.g)

        # 3D convolution; dimshuffle: last 3 dimensions must be (in, h, w)
        n_fr, h, w = input_shape
        n_fr_k, h_k, w_k = kernel_shape
        out = conv3d(
                signals=input.dimshuffle([0,2,1,3,4]), 
                filters=self.W, 
                signals_shape=(batch_size, n_fr, n_in_maps, h, w), 
                filters_shape=(n_out_maps, n_fr_k, n_in_maps, h_k, w_k),         
                border_mode='valid').dimshuffle([0,2,1,3,4])

        if use_bias: out += self.b.dimshuffle('x',0,'x','x','x')
        out = activation(out)
        # use gain after activation
        if use_gain: out *= self.g.dimshuffle('x',0,'x','x','x')

        self.output = out


class NormLayer(object):
    """ Normalization layer """

    def __init__(self, input, method="lcn", **kwargs):
        """
        method: "lcn", "gcn", "mean"

        LCN: local contrast normalization
            kwargs: 
                kernel_size=9, 
                threshold=1e-4,
                use_divisor=True

        GCN: global contrast normalization
            kwargs:
                scale=1., 
                subtract_mean=True, 
                use_std=False, 
                sqrt_bias=0., 
                min_divisor=1e-8

        MEAN: local mean subtraction
            kwargs:
                kernel_size=5
        """

        input_shape = input.shape

        # make 3D tensor out of 5D tensor -> (n_images, height, width)
        input_shape_3D = (input_shape[0]*input_shape[1]*input_shape[2], 
                            input_shape[3], input_shape[4])
        input_3D = input.reshape(input_shape_3D, ndim=3)
        if method=="lcn":
            out = self.lecun_lcn(input_3D, **kwargs)
        elif method=="gcn":
            out = self.global_contrast_normalize(input_3D,**kwargs)
        elif method=="mean":
            out = self.local_mean_subtraction(input_3D, **kwargs)
        else:
            raise NotImplementedError()

        self.output = out.reshape(input_shape)

    def lecun_lcn(self, X, kernel_size=9, threshold = 1e-4, use_divisor=True):
        """
        Yann LeCun's local contrast normalization
        Orginal code in Theano by: Guillaume Desjardins
        """

        # reshape to 4D tensor required for theano conv2d()
        X = X.dimshuffle(0,'x',1,2)

        filter_shape = (1, 1, kernel_size, kernel_size)
        filters = sharedX(gaussian_filter(kernel_size).reshape(filter_shape))
        # filters = filters.dimshuffle("x","x",0,1)

        convout = conv2d(X, filters=filters, border_mode='full')

        # For each pixel, remove mean of kernel_sizexkernel_size neighborhood
        mid = int(floor(kernel_size/2.))
        new_X = X - convout[:,:,mid:-mid,mid:-mid]

        if use_divisor:
            # Scale down norm of kernel_sizexkernel_size patch
            sum_sqr_XX = conv2d(T.sqr(X), filters=filters, border_mode='full')

            denom = T.sqrt(sum_sqr_XX[:,:,mid:-mid,mid:-mid])
            per_img_mean = denom.mean(axis=[2,3])
            divisor = T.largest(per_img_mean.dimshuffle(0,1,'x','x'), denom)
            divisor = T.maximum(divisor, threshold)

            new_X /= divisor

        return new_X

    def local_mean_subtraction(self, X, kernel_size=5):

        def mean_filter(kernel_size):
            s = kernel_size**2
            x = repeat(1./s, s).reshape((kernel_size, kernel_size))
            return x
         
        X = X.dimshuffle(0,'x',1,2)

        filter_shape = (1, 1, kernel_size, kernel_size)
        filters = sharedX(mean_filter(kernel_size).reshape(filter_shape))
        # filters = filters.dimshuffle("x","x",0,1)

        mean = conv2d(X, filters=filters, border_mode='full')
        mid = int(floor(kernel_size/2.))
        new_X = X - mean[:,:,mid:-mid,mid:-mid]  

        return new_X

    def global_contrast_normalize(self, X, scale=1., subtract_mean=True, 
        use_std=False, sqrt_bias=0., min_divisor=1e-8):

        scale = float(scale)
        ndim = X.ndim
        mean = X.mean(axis=ndim-1)
        new_X = X.copy()
        if subtract_mean:
            if X.ndim==3:
                new_X = X - mean[:,:,None]
            else: new_X = X - mean[:,:,:,None]
        if use_std:
            # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
            # Coates' code does.
            normalizers = T.sqrt(sqrt_bias + X.var(axis=ndim-1)) / scale
        else:
            normalizers = T.sqrt(sqrt_bias + (new_X ** 2).sum(axis=ndim-1)) / scale
        # Don't normalize by anything too small.
        T.set_subtensor(normalizers[(normalizers < min_divisor).nonzero()],1.)
        if X.ndim==3: new_X /= normalizers[:,:,None]
        else: new_X /= normalizers[:,:,:,None]
        return new_X


class PoolLayer(object):
    """ Subsampling and pooling layer """

    def __init__(self, input, pool_shape, use_dropout=False, dropout_p=0.5, 
        rng=RandomState(1234), method="max"):
        """
        method: "max", "avg", "L2", "L4", ...
        """

        self.__dict__.update(locals())
        del self.self

        if method=="max":
            out = max_pool_3d(input,pool_shape)
        else:
            raise NotImplementedError()

        # dropout
        if use_dropout: out = dropout(out, rng=rng, p=dropout_p)

        self.output = out


class RectificationLayer(object):
    """  Rectification layer """

    def __init__(self, input):
        self.output = T.abs_(input)