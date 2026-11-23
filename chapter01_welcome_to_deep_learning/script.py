"""Chapter 1: Welcome to Deep Learning -- code from the book, run top to bottom."""


# Where Machine Learning Runs Out of Road
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=400, noise=0.08,
                    factor=0.45, random_state=7)
print(X[:3].round(2))
print(y[:3])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7)
line = LogisticRegression().fit(X_train, y_train)
print(round(line.score(X_test, y_test), 2))

r_train = (X_train ** 2).sum(axis=1).reshape(-1, 1)
r_test = (X_test ** 2).sum(axis=1).reshape(-1, 1)
ring = LogisticRegression().fit(r_train, y_train)
print(round(ring.score(r_test, y_test), 2))


# What a Neural Network Is
from sklearn.neural_network import MLPClassifier

net = MLPClassifier(hidden_layer_sizes=(8,), max_iter=3000,
                    random_state=7)
net.fit(X_train, y_train)
print(round(net.score(X_test, y_test), 2))

sizes = [w.size for w in net.coefs_] + [b.size for b in net.intercepts_]
print(sum(sizes))
