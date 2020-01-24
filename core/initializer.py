''' 初始化变量'''

import numpy as np
import  scipy.stats as stats


def get_fans(shape):
    fan_in = shape[0] if len(shape)==2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape)==2 else shape[0]

class Initializer(object):
    def __call__(self, shape):
        return self.init(shape).astype(np.float32)
    def init(self,shape):
        raise NotImplementedError

class Constant(Initializer):
    def __init__(self,val):
        self._val = val

    def init(self,shape):
        return np.full(shape=shape,fill_value=self._val)


class XavierUniform(Initializer):
    """
    Implement the Xavier method described in
    "Understanding the difficulty of training deep feedforward neural networks"
    Glorot, X. & Bengio, Y. (2010)

    Weights will have values sampled from uniform distribution U(-a, a) where
    a = gain * sqrt(6.0 / (num_in + num_out))

    """

    def __init__(self,gain=1.0):
        self._gain = gain

    def init(self,shape):
        fan_in,fan_out = get_fans(shape)
        a = self._gain*np.sqrt(6.0/(fan_in+fan_out))
        return np.random.uniform(low = -a,high = a,size = shape)

class Zeros(Constant):
    def __init__(self,shape):
        self._val = 0.0
        self.init(shape=shape)
