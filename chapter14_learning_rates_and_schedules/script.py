"""Chapter 14: Learning Rates and Schedules -- code from the book, run top to bottom."""


# The Most Important Hyperparameter
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
loss_fn = nn.CrossEntropyLoss()

def make_net(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))

def train(learning_rate, epochs=30, seed=0):
    net = make_net(seed)
    opt = torch.optim.SGD(net.parameters(), lr=learning_rate)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    history = []
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            batch_loss = loss_fn(net(xb), yb)
            batch_loss.backward()
            opt.step()
            total += batch_loss.item() * len(xb)
        history.append(total / len(Xtr))
    with torch.no_grad():
        accuracy = (net(Xte).argmax(1) == yte).float().mean().item()
    return history, accuracy

for lr in [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 10.0]:
    history, accuracy = train(lr)
    print(lr, round(history[-1], 4), round(accuracy, 3))


# Finding a Good Learning Rate
net = make_net(0)
opt = torch.optim.SGD(net.parameters(), lr=1e-5)
loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=16, shuffle=True,
                    generator=torch.Generator().manual_seed(0))
rates, losses = [], []
lr = 1e-5
stop = False
for epoch in range(30):
    if stop:
        break
    for xb, yb in loader:
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad()
        batch_loss = loss_fn(net(xb), yb)
        batch_loss.backward()
        opt.step()
        rates.append(lr)
        losses.append(batch_loss.item())
        lr *= 1.15
        if batch_loss.item() > 8:
            stop = True
            break
print(len(rates), "steps")
print("lowest loss", round(min(losses), 4),
      "at rate", round(rates[losses.index(min(losses))], 4))


# Warmup and Decay Schedules
def train_scheduled(learning_rate, schedule=None, epochs=60, seed=0):
    net = make_net(seed)
    opt = torch.optim.SGD(net.parameters(), lr=learning_rate)
    sched = schedule(opt) if schedule else None
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    history = []
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            batch_loss = loss_fn(net(xb), yb)
            batch_loss.backward()
            opt.step()
            total += batch_loss.item() * len(xb)
        history.append(total / len(Xtr))
        if sched:
            sched.step()
    return history

step_decay = lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=20, gamma=0.1)
cosine = lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=60)

for name, schedule in [("constant", None), ("step", step_decay), ("cosine", cosine)]:
    history = train_scheduled(1.0, schedule)
    tail = history[-10:]
    print(name, round(history[-1], 4),
          round(min(tail), 4), round(max(tail), 4))


# When Training Diverges
history, accuracy = train(10.0, epochs=6)
for value in history:
    print("%.4g" % value)

Xtr_raw = torch.tensor(X_train, dtype=torch.float32)
print([round(v, 2) for v in Xtr_raw.mean(0).tolist()])
print([round(v, 2) for v in Xtr_raw.std(0).tolist()])

def train_raw(learning_rate, epochs=30, seed=0):
    net = make_net(seed)
    opt = torch.optim.SGD(net.parameters(), lr=learning_rate)
    loader = DataLoader(TensorDataset(Xtr_raw, ytr), batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            batch_loss = loss_fn(net(xb), yb)
            batch_loss.backward()
            opt.step()
            total += batch_loss.item() * len(xb)
    return total / len(Xtr_raw)

for lr in [1.0, 0.1, 0.01, 0.001]:
    print(lr, round(train_raw(lr), 4))
