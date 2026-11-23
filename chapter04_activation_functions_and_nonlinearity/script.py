"""Chapter 4: Activation Functions and Nonlinearity -- code from the book, run top to bottom."""


# Why a Network Needs a Bend
import numpy as np

x = np.array([2.0, 1.0])
W1 = np.array([[1.0, -2.0],
               [0.5,  1.0],
               [3.0,  0.0]])
b1 = np.array([0.5, -1.0, 2.0])
W2 = np.array([[1.0, 2.0, -1.0]])
b2 = np.array([0.5])

print(W2 @ (W1 @ x + b1) + b2)

W = W2 @ W1
b = W2 @ b1 + b2
print(W, b)
print(W @ x + b)

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = make_circles(n_samples=400, noise=0.08,
                    factor=0.45, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7)

for act in ["identity", "relu"]:
    net = MLPClassifier(hidden_layer_sizes=(8, 8, 8),
                        activation=act, max_iter=3000,
                        random_state=7)
    net.fit(X_train, y_train)
    print(act, round(net.score(X_test, y_test), 2))


# Sigmoid and Tanh
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.array([-6, -2, 0, 2, 6])
s = sigmoid(z)
print(s.round(3))
print((s * (1 - s)).round(3))

t = np.tanh(z)
print(t.round(3))
print((1 - t ** 2).round(3))

for layers in range(1, 7):
    print(layers, round(100 * 0.25 ** layers, 3))


# The ReLU and Why It Won
def relu(z):
    return np.maximum(0, z)

print(relu(z))
print((z > 0).astype(int))

import warnings
warnings.filterwarnings("ignore")   # hide the note that training stopped at 600 rounds

for act in ["logistic", "tanh", "relu"]:
    net = MLPClassifier(hidden_layer_sizes=(8, 8, 8),
                        activation=act, max_iter=600,
                        n_iter_no_change=600, random_state=7)
    net.fit(X_train, y_train)
    print(act, round(net.loss_curve_[-1], 3),
          round(net.score(X_test, y_test), 2))

def leaky_relu(z):
    return np.where(z > 0, z, 0.01 * z)

print(leaky_relu(z))
