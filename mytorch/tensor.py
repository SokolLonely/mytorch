import numpy as np
class Tensor:
  def __init__(self, data, requires_grad=True):
    if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
    self.data = data
    self.requires_grad = requires_grad
    self.grad = np.zeros_like(self.data)
    self._backward = lambda: None
    self._prev = set()
    self._op = ''
#    assert self.data.size == self.grad.size
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
              if self.grad is None:
                  self.grad = np.zeros_like(self.data)
              self.grad = self.grad + out.grad

          if other.requires_grad:
              if other.grad is None:
                  other.grad = np.zeros_like(other.data)
              grad = out.grad
              # #broadcasting to avoid shape mismatches
              while grad.ndim > other.data.ndim:
                  grad = grad.sum(axis=0)
              for i, dim in enumerate(other.data.shape):
                  if dim == 1:
                      grad = grad.sum(axis=i, keepdims=True)
              other.grad = other.grad + grad
      out._backward = _backward
      out._prev.add(self)
      out._prev.add(other)
      out._op = '+'
     # assert self.data.size == self.grad.size
      return out
  def __neg__(self):
    out = Tensor(-self.data)
    def _backward():
      if self.requires_grad:
        if self.grad is None or isinstance(self.grad, (float, int)):
              self.grad = np.zeros_like(self.data)
        self.grad += out.grad*(-1)
    out._backward = _backward
    out._prev.add(self)
    out._op = 'neg'
#    assert self.data.size == self.grad.size
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
          assert self.data.size == self.grad.size
          assert other.data.size == other.grad.size
        out._backward = _backward
        out._prev.add(self)
        out._prev.add(other)
        out._op = '*'
        #assert self.data.size == self.grad.size
        return out

  def __pow__(self, power):
      out = Tensor(self.data ** power, )
      def _backward():
          if self.requires_grad:
              if self.grad is None or isinstance(self.grad, (float, int)):
                  self.grad = np.zeros_like(self.data)
              self.grad += out.grad*power*(self.data ** (power - 1))
      out._backward = _backward
      out.requires_grad=self.requires_grad
      out._prev.add(self)
      out._op = '**'
      return out
  def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data @ other.data)
        def _backward():
          if self.requires_grad:
            if self.grad is None or isinstance(self.grad, (float, int)):
                self.grad = np.zeros_like(self.data)
            self.grad = (self.grad ) + out.grad @ other.data.T
            # print("matmul
          if other.requires_grad:
            if other.grad is None or isinstance(other.grad, (float, int)):
                other.grad = np.zeros_like(other.data)
            other.grad += self.data.T @ out.grad
            #assert self.data.size == self.grad.size
            #assert other.data.size == other.grad.size
        out._backward = _backward
        out._prev.add(self)
        out._prev.add(other)
        out.requires_grad = self.requires_grad or other.requires_grad
        out._op = '@'
        #assert self.data.size == self.grad.size
        return out
  def tanh(self):
    x= np.tanh(self.data)
    out = Tensor(x)
    def _backward():
      if self.requires_grad:
        if self.grad is None or isinstance(self.grad, (float, int)):
              self.grad = np.zeros_like(self.data)
        self.grad += (1 - x**2) * out.grad
        # print('tanh')
        # print(f"self.data = {self.data}")
        # print(f"out.grad = {out.grad}")
    out._backward = _backward
    out._prev.add(self)
    out._op = 'tanh'
    #assert self.data.size == self.grad.size
    return out
  def sigmoid(self):
      def internal_sigmoid(x):
          return 1 / (1 + np.exp(-x))

      x = internal_sigmoid(self.data)
      out = Tensor(x)

      def _backward():
          if self.requires_grad:
              s = internal_sigmoid(self.data)
              if self.grad is None or isinstance(self.grad, (float, int)):
                  self.grad = np.zeros_like(self.data)
              self.grad += s*(1-s) * out.grad
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
  def leaky_relu(self, negative_slope=0.01,):
      x = np.maximum(self.data*negative_slope, self.data)
      out = Tensor(x)

      def _backward():
          if self.requires_grad:
              self.grad = self.grad if self.grad is not None else np.zeros_like(x)
              grad_input = np.where(self.data > 0, 1.0, negative_slope)  # element-wise derivative
              self.grad += grad_input * out.grad

      out._backward = _backward
      out._prev.add(self)
      out._op = 'leaky_relu'
      return out
  def mean(self):
      x= self.data.mean()
      out = Tensor(x)
      def _backward():
          if self.requires_grad:
              grad = out.grad / self.data.size
              if self.grad is None:
               self.grad = np.zeros_like(self.data)
              self.grad = (self.grad) + np.ones_like(self.data) * grad
             # assert self.data.size == self.grad.size
      out._backward = _backward
      out._prev.add(self)
      out._op = 'mean'
      return out
  def softmax(self):
      def internal_softmax(x):
          exps = np.exp(x - np.max(x, axis=1, keepdims=True))
          return exps / np.sum(exps, axis=1, keepdims=True)
      x = internal_softmax(self.data)
      out = Tensor(x)

      def _backward():
          if self.requires_grad:
              if self.grad is None:
               self.grad = np.zeros_like(self.data)
              grad = out.grad  # upstream gradient
              y = out.data  # softmax output
              self.grad = (self.grad or np.zeros_like(self.data)) + \
                          y * (grad - np.sum(grad * y, axis=-1, keepdims=True))

      out._backward = _backward
      out._prev.add(self)
      out._op = 'softmax'
      return out
  def cross_entropy(self):
      def internal_softmax(x):
          exps = np.exp(x - np.max(x, axis=1, keepdims=True))
          return exps / np.sum(exps, axis=1, keepdims=True)
