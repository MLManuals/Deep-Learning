"""Chapter 15: Overfitting and Regularization -- code from the book, run top to bottom."""


# How a Network Memorizes
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=500, noise=0.25, factor=0.5, random_state=7)
X_fit, X_test, y_fit, y_test = train_test_split(
    X, y, test_size=0.6, random_state=7, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_fit, y_fit, test_size=0.5, random_state=7, stratify=y_fit)
print(len(X_train), len(X_val), len(X_test))

Xtr = torch.tensor(X_train, dtype=torch.float32)
Xva = torch.tensor(X_val, dtype=torch.float32)
Xte = torch.tensor(X_test, dtype=torch.float32)
ytr, yva, yte = torch.tensor(y_train), torch.tensor(y_val), torch.tensor(y_test)
loss_fn = nn.CrossEntropyLoss()

def build(dropout=0.0, seed=0, width=256):
    torch.manual_seed(seed)
    layers = [nn.Linear(2, width), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(width, width), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(width, 2))
    return nn.Sequential(*layers)

print(sum(p.numel() for p in build().parameters()), "parameters")

def run(dropout=0.0, weight_decay=0.0, epochs=1200, seed=0, lr=0.001):
    net = build(dropout, seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    train_hist, val_hist = [], []
    for epoch in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            batch_loss = loss_fn(net(xb), yb)
            batch_loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            train_hist.append(loss_fn(net(Xtr), ytr).item())
            val_hist.append(loss_fn(net(Xva), yva).item())
    net.eval()
    with torch.no_grad():
        accuracy = lambda Xs, ys: (net(Xs).argmax(1) == ys).float().mean().item()
        return train_hist, val_hist, accuracy(Xtr, ytr), accuracy(Xte, yte)

train_hist, val_hist, train_acc, test_acc = run()
print(round(train_hist[-1], 4), round(train_acc, 3))
print(round(val_hist[-1], 4), round(test_acc, 3))
print(round(min(val_hist), 4), val_hist.index(min(val_hist)))


# Dropout
torch.manual_seed(1)
drop = nn.Dropout(0.5)
x = torch.ones(8)
drop.train()
print([round(v, 1) for v in drop(x).tolist()])
drop.train()
print([round(v, 1) for v in drop(x).tolist()])
drop.eval()
print([round(v, 1) for v in drop(x).tolist()])

for p in [0.2, 0.5]:
    t, v, a_train, a_test = run(dropout=p)
    print(p, round(t[-1], 4), round(a_train, 3), round(v[-1], 4), round(a_test, 3))


# Weight Decay
for wd in [0.001, 0.01, 0.1]:
    t, v, a_train, a_test = run(weight_decay=wd)
    print(wd, round(t[-1], 4), round(a_train, 3), round(v[-1], 4), round(a_test, 3))

for wd in [0.0, 0.01]:
    net = build(0.0, 0)
    opt = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=wd)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
    for epoch in range(1200):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            batch_loss = loss_fn(net(xb), yb)
            batch_loss.backward()
            opt.step()
    size = sum(p.pow(2).sum().item() for p in net.parameters()) ** 0.5
    print(wd, round(size, 2))


# Early Stopping
import copy

def train_early(patience=50, max_epochs=1200, seed=0):
    net = build(0.0, seed)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best, best_state, best_epoch, waited = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            batch_loss = loss_fn(net(xb), yb)
            batch_loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = loss_fn(net(Xva), yva).item()
        if val < best:
            best, best_epoch, waited = val, epoch, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            waited += 1
            if waited >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        accuracy = (net(Xte).argmax(1) == yte).float().mean().item()
    return epoch, best_epoch, best, accuracy

stopped, best_epoch, best_val, accuracy = train_early()
print(stopped, best_epoch)
print(round(best_val, 4), round(accuracy, 3))
