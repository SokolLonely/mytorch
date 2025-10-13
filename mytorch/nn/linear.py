from .module import Module
from .. import Tensor
import numpy as np
class Linear(Module):
  def __init__(self, in_features, out_features) -> None:
     super().__init__()
     self.w = Tensor(np.random.randn(in_features, out_features) * 0.01, requires_grad=True)
     self.b = Tensor(np.zeros(out_features), requires_grad=True)
  def parameters(self):
    return [self.w, self.b]
  def __call__(self, x):
    return x @ self.w + self.b