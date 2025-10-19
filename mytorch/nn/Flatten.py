from .module import Module
from .. import Tensor
import numpy as np
class Flatten(Module):
    def __init__(self, ) -> None:
        super().__init__()

    def parameters(self):
        return []

    def __call__(self, x):
        batch_size = x.data.shape[0]
        out = Tensor(x.data.reshape(batch_size, -1), requires_grad=x.requires_grad)
        def _backward():
            x.grad = out.grad.reshape(x.data.shape)
        return out
