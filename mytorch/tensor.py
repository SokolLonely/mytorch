import numpy as np
class Tensor:
  def __init__(self, data, requires_grad=True):
    if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
    self.data = data
    self.requires_grad = requires_grad
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set()
    self._op = ''
  def __repr__(self):
        return f"Value(data={self.data})"
  def backward(self):
      if not self.requires_grad:
            #("if")
            return
      topo = []
      visited = set()
      def build_topo(v):
        if v not in visited:
          visited.add(v)
          for child in v._prev:
            build_topo(child)
          topo.append(v)
      build_topo(self)
      self.grad = np.ones_like(self.data)
      for node in reversed(topo):
        node._backward()
        #print(node)
  #adds tensor
  def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data + other.data)
        def _backward():
          if self.requires_grad:
            self.grad = (self.grad or 0) + out.grad
          if other.requires_grad:
            other.grad = (other.grad or 0) + out.grad
        out._backward = _backward
        out._prev.add(self)
        out._prev.add(other)
        out._op = '+'
        return out
  def __neg__(self):
    out = Tensor(-self.data)
    def _backward():
      if self.requires_grad:
        self.grad = (self.grad or 0)+ out.grad*(-1)
    out._backward = _backward
    out._prev.add(self)
    out._op = 'neg'
    return out
  def __radd__(self, other): # other + self
        return self + other


  def __sub__(self, other): # self - other
        return self + (-other)


  def __rsub__(self, other): # other - self
        return other + (-self)


  def __rmul__(self, other): # other * self
        return self * other
  def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data * other.data)
        def _backward():
          if self.requires_grad:
            self.grad += other.data * out.grad
            # print("mul")
            # print(f"self.data = {self.data}")
            # print(f"other.data = {other.data}")
            # print(f"out.grad = {out.grad}")
          if other.requires_grad:
            other.grad += self.data * out.grad
            # print(f"self.data = {self.data}")
            # print(f"other.data = {other.data}")
            # print(f"out.grad = {out.grad}")
        out._backward = _backward
        out._prev.add(self)
        out._prev.add(other)
        out._op = '*'
        return out
  def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data @ other.data)
        def _backward():
          if self.requires_grad:
            self.grad = (self.grad or 0) + out.grad @ other.data.T
            # print("matmul
          if other.requires_grad:
            other.grad = (other.grad or 0) + self.data.T @ out.grad
        out._backward = _backward
        out._prev.add(self)
        out._prev.add(other)
        out.requires_grad = self.requires_grad or other.requires_grad
        out._op = '@'
        return out
  def tanh(self):
    x= np.tanh(self.data)
    out = Tensor(x)
    def _backward():
      if self.requires_grad:
        self.grad = (self.grad or 0) + (1 - x**2) * out.grad
        # print('tanh')
        # print(f"self.data = {self.data}")
        # print(f"out.grad = {out.grad}")
    out._backward = _backward
    out._prev.add(self)
    out._op = 'tanh'
    return out
  def sigmoid(self):
      def internal_sigmoid(x):
          return 1 / (1 + np.exp(-x))

      x = internal_sigmoid(self.data)
      out = Tensor(x)

      def _backward():
          if self.requires_grad:
              s = internal_sigmoid(self.data)
              self.grad = (self.grad or 0) + s*(1-s) * out.grad
      out._backward = _backward
      out._prev.add(self)
      out._op = 'sigmoid'
      return out
  def relu(self):
      x = np.maximum(0, self.data)
      out = Tensor(x)

      def _backward():
          if self.requires_grad:
              self.grad = self.grad if self.grad is not None else np.zeros_like(x)
              self.grad += (x > 0).astype(x.dtype) * out.grad
      out._backward = _backward
      out._prev.add(self)
      out._op = 'relu'
      return out