import numpy as np
from ... import Tensor
def cross_entropy( logits :Tensor, targets:Tensor):
    def internal_softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    probs = internal_softmax(logits.data)
    n = logits.data.shape[0]

    log_likelihood = -np.log(probs[range(n), targets.data.astype(int)])
    loss_value = log_likelihood.mean()
    out = Tensor(loss_value, requires_grad=logits.requires_grad)
    def _backward():
        grad = probs
        grad[range(n), targets.data.astype(int)]-=1
        grad /=n
        logits.grad += grad

    out._backward = _backward
    out._prev.add(logits)
    out._op = 'cross_entropy'
    return out