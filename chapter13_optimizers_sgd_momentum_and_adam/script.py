"""Chapter 13: Optimizers: SGD, Momentum and Adam -- code from the book, run top to bottom."""


# Plain SGD and Its Problems
def loss(x, y):
    return 0.1 * x ** 2 + 2 * y ** 2

def slopes(x, y):
    return 0.2 * x, 4 * y

def descend(learning_rate, beta=0.0, steps=20):
    x, y = -9.0, 2.0
    vx = vy = 0.0
    path = [(x, y)]
    for _ in range(steps):
        gx, gy = slopes(x, y)
        vx = beta * vx + gx
        vy = beta * vy + gy
        x -= learning_rate * vx
        y -= learning_rate * vy
        path.append((x, y))
    return path

plain = descend(0.4)
print([round(y, 2) for x, y in plain[:6]])
print([round(x, 2) for x, y in plain[:6]])

print(round(plain[-1][0], 3), round(loss(*plain[-1]), 4))


# Momentum
moving = descend(0.4, beta=0.5)
print(round(moving[-1][0], 3), round(loss(*moving[-1]), 4))

import torch
import torch.nn as nn

model = nn.Linear(4, 3)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
group = opt.param_groups[0]
print(group["lr"], group["momentum"])


# Adam and the Adaptive Family
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3,
    random_state=7, stratify=iris.target)
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
Xtr = torch.tensor((X_train - mean) / std, dtype=torch.float32)
Xte = torch.tensor((X_test - mean) / std, dtype=torch.float32)
ytr, yte = torch.tensor(y_train), torch.tensor(y_test)
train_ds = TensorDataset(Xtr, ytr)
loss_fn = nn.CrossEntropyLoss()

def run(make_optimizer, epochs=30):
    torch.manual_seed(7)
    net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
    optimizer = make_optimizer(net)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    torch.manual_seed(7)
    for epoch in range(epochs):
        for xb, yb in loader:
            l = loss_fn(net(xb), yb)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
    with torch.no_grad():
        return (loss_fn(net(Xtr), ytr).item(),
                (net(Xte).argmax(1) == yte).float().mean().item())

choices = [
    ("SGD", lambda n: torch.optim.SGD(n.parameters(), lr=0.1)),
    ("momentum", lambda n: torch.optim.SGD(n.parameters(), lr=0.1,
                                           momentum=0.9)),
    ("Adam", lambda n: torch.optim.Adam(n.parameters(), lr=0.01))]

for name, make in choices:
    final, acc = run(make)
    print(name, round(final, 4), round(acc, 3))


# Which Optimizer to Reach For
for lr in [0.1, 0.01, 0.001]:
    final, acc = run(lambda n: torch.optim.Adam(n.parameters(), lr=lr))
    print(lr, round(final, 4), round(acc, 3))

for lr in [0.1, 0.01, 0.001]:
    final, acc = run(lambda n: torch.optim.Adam(n.parameters(), lr=lr),
                     epochs=300)
    print(lr, round(final, 4), round(acc, 3))
