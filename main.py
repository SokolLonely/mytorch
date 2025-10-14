#file with example usage and test cases
from mytorch.nn import Linear
from mytorch.tensor import Tensor
from mytorch.optim import SGD
from mytorch.nn import Module
import numpy as np
class TwoLayerNet(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, output_size)

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

    def __call__(self, x):
        # Simple network: Linear -> Tanh -> Linear
        x = self.fc1(x).tanh()   # activation after first layer
        x = self.fc2(x)          # output layer (no activation if regression)
        return x
x = Tensor([[1, 2]], requires_grad=False)
y = Tensor([[3]], requires_grad=False)

linear = Linear(2, 1)
opt = SGD(linear.parameters(), lr=0.01)

for _ in range(10):
    pred = linear(x)
    loss = (pred - y) * (pred - y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(loss.data)
print(pred)
print("==============")
x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)  # 2 samples, 2 features
y = Tensor([[3.0], [7.0]], requires_grad=False)            # targets
net = TwoLayerNet(input_size=2, hidden_size=4, output_size=1)
opt = SGD(net.parameters(), lr=0.01)
print("starting loop")
for epoch in range(500):
    pred = net(x)
    # Mean Squared Error
    loss = ((pred - y) * (pred - y)).mean()

    # backward pass
    loss.backward()

    # update parameters
    opt.step()
    opt.zero_grad()

    print(f"Epoch {epoch + 1}: loss = {loss.data.item()}")

print("Final predictions:")
print(pred.data)
# --- TEST 1: Simple addition ---
# a = Tensor(2.0, requires_grad=True)
# b = Tensor(3.0, requires_grad=True)
# d = Tensor(3.0, requires_grad=True)
# c = a + b+d
# print("Forward (a + b):", c.data)  # Expected: 5.0
# c.backward()
# print("Grad a:", a.grad)           # Expected: 1
# print("Grad b:", b.grad)           # Expected: 1
# print("Grad d:", d.grad)
# print()

# --- TEST 2: Simple multiplication ---
# a = Tensor(2.0, requires_grad=True)
# b = Tensor(-3.0, requires_grad=True)
# c = Tensor(10.0, requires_grad=True)
#
# e = a * b + c
# print("Forward result (a*b + c):", e.data)  # expected 4.0
#
# e.backward()
#
# print("Grad a (should be b=-3):", a.grad)
# print("Grad b (should be a=2):", b.grad)
# print("Grad c (should be 1):", c.grad)

# --- TEST 3: Combination of operations ---
# a = Tensor(2.0, requires_grad=True)
# b = Tensor(-3.0, requires_grad=True)
# c = Tensor(10.0, requires_grad=True)
# e = (a * b)+c
# d = e.sigmoid()
# # print("Forward result (a*b + c):", e.data)
# d.backward()
# print("Grad a:", a.grad)
# print("Grad b:", b.grad)
# print("Grad c:", c.grad)
# print("Grad d:", d.grad)
# print("Grad e:", e.grad)

# # --- TEST 4: Vector test (NumPy array input) ---
# a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
# b = Tensor(np.array([2.0, 0.5, -1.0]), requires_grad=True)
# c = (a + b).tanh()
# print("Forward vector tanh(a+b):", c.data)
# c.backward()
# print("Grad a:", a.grad)
# print("Grad b:", b.grad)
# x = Tensor(np.array([-1.0, 0.0, 2.0]), requires_grad=True)
# y = x.leaky_relu()  # ReLU applied element-wise
#
# # Assume we start backward with gradient 1 for each element
# y.grad = np.ones_like(y.data)
# y._backward()  # call backward manually if needed
#
# print("Forward result y.data:", y.data)
# # Expected: [0.0, 0.0, 2.0]
#
# print("Grad x:", x.grad)
# Expected derivative: [0, 0, 1] element-wise