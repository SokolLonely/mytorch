from .module import Module
from .. import Tensor
import numpy as np
class Conv2D(Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0) -> None:
     super().__init__()
     if isinstance(kernel_size, int):
         kernel_size = (kernel_size, kernel_size)
     self.stride=str
     self.padding=padding
     self.w = Tensor(np.random.randn(in_channels, out_channels, kernel_size) * 0.01, requires_grad=True)
     self.b = Tensor(np.zeros(out_channels), requires_grad=True)
  def parameters(self):
    return [self.w, self.b]
  def __call__(self, x):
    return x @ self.w + self.b