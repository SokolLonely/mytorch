from mytorch.nn import Linear
from mytorch.tensor import Tensor
from mytorch.optim import SGD
x = Tensor([[1, 2]], requires_grad=False)
y = Tensor([[3]], requires_grad=False)

linear = Linear(2, 1)
opt = SGD(linear.parameters(), lr=0.1)

for _ in range(10):
    pred = linear(x)
    loss = (pred - y) * (pred - y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(loss.data)
print(pred)
# Paste your Tensor class definition here before running this test
# (or assume it's already defined in the same file)

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
# d = e.tanh()
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
