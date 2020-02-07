import numpy as np
class BaseLoss(object):
    def loss(self,predicted,actual):
        raise NotImplementedError

    def grad(self,predicted,actual):
        raise NotImplementedError

class CrossEntropyLoss(BaseLoss):
    def loss(self,predicted,actual):
        m = predicted.shape[0]
        exps = np.exp()
