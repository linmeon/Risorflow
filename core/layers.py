import numpy as np

from core.initializer import XavierUniform
from core.initializer import Zeros


class Layer(object):
    def __init__(self, name):
        self.name = name
        self.params, self.grads = dict(), dict()

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self,
                 num_in,
                 num_out,
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__('Dense')

        self.params = {
            "w": w_init([num_in, num_out]),
            "b": b_init([1, num_out])}

        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad):
        self.grads['w'] = self.inputs.T @ grad
        self.grads['b'] = np.sum(grad, axis=0)
        return grad @ self.params['w'].T


class Activation(Layer):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def backward(self, grad):
        return self.derivative_func(self.inputs) * grad

    def func(self, x):
        raise NotImplementedError

    def derivative_func(self, x):
        raise NotImplementedError


class Sigmoid(Activation):
    def __init__(self):
        super().__init__('Sigmoid')

    def func(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative_func(self, x):
        return self.func(x) * (1.0 - self.func(x))


class ReLU(Activation):
    def __init__(self):
        super().__init__('ReLU')

    def func(self, x):
        return np.max(0.0, x)

    def derivative_func(self, x):
        return x > 0.0
