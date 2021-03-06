''' 初始化变量'''

import numpy as np


def get_fans(shape):
    fan_in = shape[0] if len(shape)==2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape)==2 else shape[0]
    return fan_in, fan_out


class Initializer(object):
    def __call__(self, shape):
        return self.init(shape).astype(np.float32)

    def init(self,shape):
        raise NotImplementedError


class Zeros(Initializer):
    def init(self,shape):
        return np.zeros(shape)


class XavierUniform(Initializer):
    """
    Implement the Xavier method described in
    "Understanding the difficulty of training deep feed forward neural networks"
    Glorot, X. & Bengio, Y. (2010)

    Weights will have values sampled from uniform distribution U(-a, a) where
    a = gain * sqrt(6.0 / (num_in + num_out))

    """

    def __init__(self, gain=1.0):
        self._gain = gain

    def init(self, shape):
        fan_in, fan_out = get_fans(shape)
        a = self._gain*np.sqrt(6.0/(fan_in+fan_out))
        return np.random.uniform(low=-a, high=a, size=shape)


class XavierNormal(Initializer):
    """
    Implement the Xavier method described in
    "Understanding the difficulty of training deep feed forward neural networks"
    Glorot, X. & Bengio, Y. (2010)

    Weights will have values sampled from normal distribution U(-a, a) where
    a = gain * sqrt(6.0 / (num_in + num_out))

    """

    def __init__(self, gain=1.0):
        self._gain = gain

    def init(self, shape):
        fan_in, fan_out = get_fans(shape)
        std = self._gain * np.sqrt(6.0 / (fan_in + fan_out))
        np.random.normal(loc=0.0, scale=std, size=shape)


class HeUniform(Initializer):
    """
    Implement the He initialization method described in
    “Delving deep into rectifiers: Surpassing human-level performance
    on ImageNet classification” He, K. et al. (2015)
    Weights will have values sampled from uniform distribution U(-a, a) where
    a = sqrt(6.0 / num_in)
    """

    def __init__(self, gain=1.0):
        self._gain = gain

    def init(self, shape):
        fan_in, _ = get_fans(shape)
        a = self._gain * np.sqrt(6.0 / fan_in)
        return np.random.uniform(low=-a, high=a, size=shape)


class HeNormal(Initializer):
    """
    Implement the He initialization method described in
    “Delving deep into rectifiers: Surpassing human-level performance
    on ImageNet classification” He, K. et al. (2015)
    Weights will have values sampled from normal distribution N(0, std) where
    std = sqrt(2.0 / num_in)
    """

    def __init__(self, gain=1.0):
        self._gain = gain

    def init(self, shape):
        fan_in, _ = get_fans(shape)
        std = self._gain * np.sqrt(2.0 / fan_in)
        return np.random.normal(loc=0.0, scale=std, size=shape)
