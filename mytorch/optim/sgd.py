class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
          if p.requires_grad:
            temp = self.lr * p.grad.reshape(p.data.shape)
            p.data -= temp
    def zero_grad(self):
        for p in self.parameters:
            p.grad = None