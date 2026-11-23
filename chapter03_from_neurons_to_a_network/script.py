"""Chapter 3: From Neurons to a Network -- code from the book, run top to bottom."""


# Stacking Neurons into a Layer
import numpy as np

x = np.array([8, 3])            # hours studied, hours slept
W = np.array([[0.9, 0.1],
              [0.1, 0.9],
              [0.5, 0.5]])
b = np.array([-4.0, -4.0, -5.0])

z = W @ x + b
print(z)
print((z > 0).astype(int))

print(W.shape, x.shape, z.shape)


# Hidden Layers and Depth
sizes = [2, 3, 3, 1]
total = 0
for n_in, n_out in zip(sizes[:-1], sizes[1:]):
    count = n_out * n_in + n_out
    print(n_in, "to", n_out, ":", count)
    total += count
print(total)


# The Forward Pass by Hand
def step(z):
    return (z > 0).astype(int)

W1 = np.array([[ 1.0,  1.0],
               [-1.0, -1.0]])
b1 = np.array([-0.5, 1.5])
W2 = np.array([1.0, 1.0])
b2 = -1.5

def forward(x):
    h = step(W1 @ x + b1)
    y = step(W2 @ h + b2)
    return h, y

for pair in ([0, 0], [0, 1], [1, 0], [1, 1]):
    h, y = forward(np.array(pair))
    print(pair, h, y)


# Why One Layer Cannot Solve XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([0, 1, 1, 0])
rng = np.random.default_rng(0)

best = 0
for trial in range(100_000):
    w = rng.uniform(-5, 5, 2)
    bias = rng.uniform(-5, 5)
    guess = ((X @ w + bias) > 0).astype(int)
    best = max(best, (guess == target).mean())
print(best)
