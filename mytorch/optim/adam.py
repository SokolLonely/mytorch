import numpy as np
class Adam:
    def __init__(self, parameters, a=0.01, b1 = 0.9, b2 = 0.999, eps = 1e-8):
        self.parameters = list(parameters)
        self.a = a
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0
    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
          if p.requires_grad:
            grad = p.grad.reshape(p.data.shape)
            self.m[i] = self.m[i] * self.b1 + (1-self.b1)*grad
            self.v[i] = self.v[i] * self.b2 + (1-self.b2)*(grad**2)
            m_hat = self.m[i] / (1-self.b1**self.t)
            v_hat = self.v[i] / (1-self.b2**self.t)
            p.data -= self.a *m_hat/(np.sqrt(v_hat)+self.eps)
    def zero_grad(self):
        for p in self.parameters:
            p.grad = None