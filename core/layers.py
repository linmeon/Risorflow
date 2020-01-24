from core.initializer import XavierUniform
from core.initializer import Zeros


class Layer(object):
    def __init__(self, name):
        self.params = {p: None for p in self.param_names}
        self.ut_params = {p: None for p in self.ut_param_names}

        self.grads = {}
        self.shapes = {}

        self.is_training = True
        self.is_init = False

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def set_phase(self, phase):
        self.is_training = True if phase == "TRAIN" else False

    @property
    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        shape = None if not self.shapes else self.shapes
        return 'layer: s% \t shape: %s' % (self.name, shape)

    @property
    def paranames(self):
        return ()

    @property
    def ut_param_names(self):
        return ()


class Dense(Layer):
    def __init__(self,
                 num_out,
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super.__init__()
        self.initializers = {'w': w_init, 'b': b_init}
        self.shapes = {'w': [None, num_out], 'b': [num_out]}
        self.inputs = None

    def forward(self, inputs):
        if not self.is_init:
            self.shapes['w'][0] = inputs.shape[1]
            self.inputs = inputs
        return inputs@self.params['w']+self.params['b']
    def backward(self, grad):


    def _init_params(self):
        for p in self.param_names:
            self.params[p] = self.initializers[p](self.shapes[p])
        self.is_init = True


    @property
    def param_names(self):
        return 'w', 'b'
