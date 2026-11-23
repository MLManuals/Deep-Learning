"""Chapter 12: Datasets, DataLoaders and Batching -- code from the book, run top to bottom."""


# The Dataset and the DataLoader
import torch
import torch.nn as nn
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
ytr = torch.tensor(y_train)
yte = torch.tensor(y_test)

train_ds = TensorDataset(Xtr, ytr)
print(len(train_ds))
print(train_ds[0][0].numpy().round(2), train_ds[0][1].item())

loader = DataLoader(train_ds, batch_size=16, shuffle=True)
print(len(loader))

torch.manual_seed(7)
for step, (xb, yb) in enumerate(loader):
    print(step, tuple(xb.shape), yb[:5].numpy())
    if step == 1:
        break


# Shuffling, Epochs and Steps
import math

for bs in [1, 8, 16, 32, 105]:
    print(bs, math.ceil(105 / bs), len(DataLoader(train_ds, batch_size=bs)))


# Stochastic versus Full Batch
loss_fn = nn.CrossEntropyLoss()

def train(batch_size, epochs=30, lr=0.1):
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    updates = 0
    for epoch in range(epochs):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates += 1
    return model, updates

torch.manual_seed(7)
tiny = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
opt = torch.optim.SGD(tiny.parameters(), lr=0.1)
one_at_a_time = DataLoader(train_ds, batch_size=1, shuffle=True)

seen = []
for xb, yb in one_at_a_time:
    loss = loss_fn(tiny(xb), yb)
    seen.append(round(loss.item(), 2))
    opt.zero_grad()
    loss.backward()
    opt.step()
print(seen[:8])


# How Batch Size Changes Training
for bs in [1, 8, 32, 105]:
    model, updates = train(bs)
    with torch.no_grad():
        train_loss = loss_fn(model(Xtr), ytr).item()
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    print(bs, updates, round(train_loss, 4), round(acc, 3))
