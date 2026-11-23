"""Chapter 8: Tensors in PyTorch -- code from the book, run top to bottom."""


# Meet PyTorch
import torch

print(torch.__version__)


# The Tensor, a Smarter Array
t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print(t)
print(t.shape, t.dtype)

print(torch.tensor([1, 2, 3]).dtype)
print(torch.tensor([1.0, 2.0, 3.0]).dtype)

torch.manual_seed(7)
print(torch.zeros(2, 3))
print(torch.arange(0, 10, 2))
print(torch.randn(2, 3).round(decimals=3))


# Everyday Tensor Operations
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([10.0, 20.0, 30.0])
print(x + y, x * y, x * 2)

m = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
print(m + torch.tensor([10.0, 100.0]))

W = torch.tensor([[0.9, 0.1],
                  [0.1, 0.9],
                  [0.5, 0.5]])
student = torch.tensor([8.0, 3.0])
print(W @ student)

r = torch.arange(6.0)
print(r)
print(r.reshape(2, 3))
print(r.reshape(2, 3).T)

s = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
print(s.sum(), s.mean(), s.sum(dim=0), s.max())


# From numpy to Tensors
import numpy as np

arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)
print(t, t.dtype)

arr[0] = 99.0
print(t)

arr2 = np.array([1.0, 2.0, 3.0])
t2 = torch.tensor(arr2, dtype=torch.float32)
arr2[0] = 99.0
print(t2)

def step(z):
    return (z > 0).float()

W1 = torch.tensor([[ 1.0,  1.0],
                   [-1.0, -1.0]])
b1 = torch.tensor([-0.5, 1.5])
W2 = torch.tensor([1.0, 1.0])
b2 = torch.tensor(-1.5)

for pair in ([0., 0.], [0., 1.], [1., 0.], [1., 1.]):
    x = torch.tensor(pair)
    h = step(W1 @ x + b1)
    y = step(W2 @ h + b2)
    print(pair, h.numpy(), y.item())
