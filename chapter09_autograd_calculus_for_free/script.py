"""Chapter 9: Autograd: Calculus for Free -- code from the book, run top to bottom."""


# The Array That Remembers
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
print(y)
print(x.grad)

y.backward()
print(x.grad)


# How Autograd Builds a Graph
a = torch.tensor(2.0, requires_grad=True)
b = a * 3
c = b ** 2
print(type(b.grad_fn).__name__, type(c.grad_fn).__name__)

c.backward()
print(a.grad)


# Reproducing Your Hand Derived Gradients
W1 = torch.tensor([[0.5, -0.5],
                   [0.5,  0.5]], requires_grad=True)
b1 = torch.tensor([0.0, 0.0], requires_grad=True)
W2 = torch.tensor([1.0, -1.0], requires_grad=True)
b2 = torch.tensor(0.0, requires_grad=True)

x = torch.tensor([1.0, 0.0])
target = torch.tensor(1.0)

h = torch.sigmoid(W1 @ x + b1)
y = torch.sigmoid(W2 @ h + b2)
loss = (y - target) ** 2
print(h.detach().numpy().round(4), round(y.item(), 4), round(loss.item(), 4))

loss.backward()
print(W2.grad.numpy().round(4), round(b2.grad.item(), 4))

print(W1.grad.numpy().round(4))
print(b1.grad.numpy().round(4))

temp = torch.tensor([20., 24., 28., 30., 33., 35.])
drinks = torch.tensor([41., 47., 58., 59., 67., 69.])

w = torch.tensor(1.5, requires_grad=True)
mse = ((w * temp - drinks) ** 2).mean()
mse.backward()
print(round(w.grad.item(), 2))


# When to Turn Autograd Off
w = torch.tensor(1.0, requires_grad=True)
for i in range(3):
    loss = w ** 2
    loss.backward()
    print(i, w.grad.item())

w = torch.tensor(1.0, requires_grad=True)
for i in range(3):
    if w.grad is not None:
        w.grad.zero_()
    loss = w ** 2
    loss.backward()
    print(i, w.grad.item())

p = torch.tensor(2.0, requires_grad=True)
tracked = p * 5
with torch.no_grad():
    untracked = p * 5
print(tracked.requires_grad, untracked.requires_grad)

print(tracked.detach().requires_grad)

w = torch.tensor(1.5, requires_grad=True)
for step in range(31):
    loss = ((w * temp - drinks) ** 2).mean()
    if step in (0, 1, 2, 5, 10, 30):
        print(step, round(w.item(), 4), round(loss.item(), 3))
    loss.backward()
    with torch.no_grad():
        w -= 0.0003 * w.grad
    w.grad.zero_()
