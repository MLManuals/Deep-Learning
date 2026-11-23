"""Chapter 6: Gradient Descent From Scratch -- code from the book, run top to bottom."""


# The Idea of Rolling Downhill
def loss(w):
    return (w - 2) ** 2

def slope(w):
    return 2 * (w - 2)

w = 5.0
for step in range(1, 7):
    g = slope(w)
    w = w - 0.3 * g
    print(step, round(g, 3), round(w, 4), round(loss(w), 4))


# Following the Slope
import numpy as np

temp = np.array([20, 24, 28, 30, 33, 35])
drinks = np.array([41, 47, 58, 59, 67, 69])

def mse(w):
    return ((w * temp - drinks) ** 2).mean()

h = 0.001
measured = (mse(1.5 + h) - mse(1.5 - h)) / (2 * h)
print(round(measured, 2))

def slope_mse(w):
    return (2 * temp * (w * temp - drinks)).mean()

print(round(slope_mse(1.5), 2), round(slope_mse(2.4), 2))


# The Learning Rate, the One Dial That Matters
def descend(start, learning_rate, steps):
    w = start
    path = [w]
    for _ in range(steps):
        w = w - learning_rate * slope_mse(w)
        path.append(w)
    return path

for lr in [0.00005, 0.0003, 0.0013]:
    final = descend(1.5, lr, 30)[-1]
    print(lr, round(final, 4), round(mse(final), 3))

for step, w in enumerate(descend(1.5, 0.0013, 5)):
    print(step, round(float(w), 3))


# Watching It Converge
path = descend(1.5, 0.0003, 30)
losses = [mse(w) for w in path]
for step in [0, 1, 2, 5, 10, 30]:
    print(step, round(path[step], 4), round(losses[step], 3))


# Why the Hill Is Not Smooth: Local Minima and Saddle Points
def hilly(w):
    return w ** 4 - 8 * w ** 2 + 2 * w + 21

def hilly_slope(w):
    return 4 * w ** 3 - 16 * w + 2

for start in [-1.0, 1.0]:
    w = start
    for _ in range(60):
        w = w - 0.02 * hilly_slope(w)
    print(start, round(w, 4), round(hilly(w), 3))
