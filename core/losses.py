import numpy as np


class BaseLoss(object):
    def loss(self,predicted,actual):
        raise NotImplementedError

    def grad(self,predicted,actual):
        raise NotImplementedError


class CrossEntropyLoss(BaseLoss):
    def loss(self,predicted,actual):
        m = predicted.shape[0]
        return 0.5 * np.sqrt((predicted - actual) ** 2) / m

    def grad(self, predicted, actual):
        m = predicted.shape[0]
        return (predicted - actual) / m


class SoftmaxCrossEntropyLoss(BaseLoss):
    def loss(self, predicted, actual):
        m = predicted.shape[0]
        # Softmax
        exps = np.exp(predicted - np.max(predicted))
        p = exps / np.sum(exps)
        # cross entropy loss
        nll = -np.log(np.sum(p * actual), axis=1)
        return np.sum(nll) / m

    def grad(self, predicted, actual):
        m = predicted.shape[0]
        grad = np.copy(predicted)
        grad -= actual
        return grad / m
