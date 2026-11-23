"""Chapter 2: The Artificial Neuron -- code from the book, run top to bottom."""


# Weights, Bias and the Weighted Sum
import numpy as np

student = np.array([6, 7])        # hours studied, hours slept
weights = np.array([0.6, 0.4])
bias = -5.0

z = np.dot(weights, student) + bias
print(round(z, 2))


# The Activation Step
def step(z):
    return 1 if z > 0 else 0

print(step(1.4), step(-0.6), step(-0.2))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for z in [1.4, -0.6, -0.2]:
    print(round(sigmoid(z), 3))


# A Neuron in a Few Lines of numpy
def neuron(x, weights, bias):
    z = np.dot(weights, x) + bias
    return step(z)

print(neuron(np.array([6, 7]), weights, bias))
print(neuron(np.array([2, 8]), weights, bias))

rng = np.random.default_rng(3)
study = rng.uniform(0, 10, 60).round(1)
sleep = rng.uniform(3, 10, 60).round(1)
X = np.column_stack([study, sleep])
noise = rng.normal(0, 0.6, 60)
passed = ((0.6 * study + 0.4 * sleep - 5 + noise) > 0).astype(int)
print(X[:3])
print(passed[:3])

z = X @ weights + bias
guess = (z > 0).astype(int)
print(guess[:3])
print((guess == passed).mean().round(2))
