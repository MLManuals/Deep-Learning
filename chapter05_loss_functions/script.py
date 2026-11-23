"""Chapter 5: Loss Functions -- code from the book, run top to bottom."""


# Measuring How Wrong a Network Is
import numpy as np

actual = np.array([42, 35, 50, 28, 46])    # cups of tea sold
guess = np.array([47, 30, 50, 28, 46])

errors = guess - actual
print(errors)
print(errors.sum())


# Mean Squared Error for Numbers
guess_a = np.array([40, 38, 47, 30, 44])
guess_b = np.array([42, 35, 50, 28, 35])

for g in (guess_a, guess_b):
    err = g - actual
    print(np.abs(err).mean(), (err ** 2).mean())

temp = np.array([20, 24, 28, 30, 33, 35])
drinks = np.array([41, 47, 58, 59, 67, 69])

def mse(w):
    return ((w * temp - drinks) ** 2).mean()

for w in [1.5, 1.8, 2.0, 2.1, 2.5]:
    print(w, round(mse(w), 1))


# Cross Entropy for Classes
p = np.array([0.99, 0.9, 0.6, 0.5, 0.1, 0.01])
print((-np.log(p)).round(3))
print(((1 - p) ** 2).round(3))

def softmax(scores):
    e = np.exp(scores)
    return e / e.sum()

scores = np.array([2.0, 1.0, 0.1])    # cat, dog, rabbit
probs = softmax(scores)
print(probs.round(3))

print(round(-np.log(probs[0]), 3), round(-np.log(probs[2]), 3))


# Reading a Loss Value
print(round(-np.log(1 / 2), 3), round(-np.log(1 / 10), 3))

baseline = ((drinks - drinks.mean()) ** 2).mean()
print(round(baseline, 1), round(mse(2.0), 1))
