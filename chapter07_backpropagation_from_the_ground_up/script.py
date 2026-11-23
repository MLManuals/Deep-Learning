"""Chapter 7: Backpropagation From the Ground Up -- code from the book, run top to bottom."""


# Passing the Error Backward
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

W1 = np.array([[0.5, -0.5],
               [0.5,  0.5]])
b1 = np.array([0.0, 0.0])
W2 = np.array([1.0, -1.0])
b2 = 0.0

x = np.array([1.0, 0.0])
target = 1.0

z1 = W1 @ x + b1
h = sigmoid(z1)
z2 = W2 @ h + b2
y = sigmoid(z2)
loss = (y - target) ** 2
print(z1.round(4), h.round(4))
print(round(z2, 4), round(y, 4), round(loss, 4))

dz2 = 2 * (y - target) * y * (1 - y)
print(round(dz2, 4))

grad_W2 = dz2 * h
grad_b2 = dz2
print(grad_W2.round(4), round(grad_b2, 4))

dh = dz2 * W2
dz1 = dh * h * (1 - h)
print(dh.round(4), dz1.round(4))

grad_W1 = np.outer(dz1, x)
grad_b1 = dz1
print(grad_W1.round(4))

def loss_with(W1):
    y_ = sigmoid(W2 @ sigmoid(W1 @ x + b1) + b2)
    return (y_ - target) ** 2

step = 1e-6
up, down = W1.copy(), W1.copy()
up[0, 0] += step
down[0, 0] -= step
measured = (loss_with(up) - loss_with(down)) / (2 * step)
print(round(measured, 6), round(grad_W1[0, 0], 6))


# Training a Tiny Network by Hand
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
T = np.array([0, 1, 1, 0], dtype=float)

rng = np.random.default_rng(3)
W1 = rng.normal(0, 1, (2, 2))
b1 = np.zeros(2)
W2 = rng.normal(0, 1, 2)
b2 = 0.0

history = []
for epoch in range(4001):
    H = sigmoid(X @ W1.T + b1)
    Y = sigmoid(H @ W2 + b2)
    history.append(((Y - T) ** 2).mean())

    dZ2 = 2 * (Y - T) * Y * (1 - Y) / len(X)
    dZ1 = np.outer(dZ2, W2) * H * (1 - H)

    W2 -= 0.5 * (dZ2 @ H)
    b2 -= 0.5 * dZ2.sum()
    W1 -= 0.5 * (dZ1.T @ X)
    b1 -= 0.5 * dZ1.sum(axis=0)

for epoch in [0, 500, 1000, 2000, 4000]:
    print(epoch, round(history[epoch], 4))

print(sigmoid(sigmoid(X @ W1.T + b1) @ W2 + b2).round(2))


# Why Backprop Scales to Millions of Weights
for n_in, n_hidden in [(2, 2), (20, 20), (784, 128)]:
    weights = n_hidden * n_in + n_hidden + n_hidden + 1
    print(weights, 2 * weights, 2)
