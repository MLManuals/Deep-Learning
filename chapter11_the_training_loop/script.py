"""Chapter 11: The Training Loop -- code from the book, run top to bottom."""


# Writing the Loop
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.data.shape)
print(iris.feature_names)
print(iris.target_names)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3,
    random_state=7, stratify=iris.target)
print(X_train.shape, X_test.shape)

import torch
import torch.nn as nn

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
print(mean.round(2), std.round(2))

Xtr = torch.tensor((X_train - mean) / std, dtype=torch.float32)
Xte = torch.tensor((X_test - mean) / std, dtype=torch.float32)
ytr = torch.tensor(y_train)
yte = torch.tensor(y_test)
print(Xtr.dtype, ytr.dtype)

torch.manual_seed(7)
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3))
print(sum(p.numel() for p in model.parameters()))

loss_fn = nn.CrossEntropyLoss()
scores = torch.tensor([[2.0, 1.0, 0.1]])
target = torch.tensor([0])
print(round(loss_fn(scores, target).item(), 4))
print(round(loss_fn(torch.softmax(scores, dim=1), target).item(), 4))

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

history = []
for epoch in range(301):
    pred = model(Xtr)
    loss = loss_fn(pred, ytr)
    history.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Your First Trained Network, a Full Multi Class Classification Walkthrough
for epoch in [0, 50, 100, 200, 300]:
    print(epoch, round(history[epoch], 4))

with torch.no_grad():
    train_acc = (model(Xtr).argmax(1) == ytr).float().mean()
    test_out = model(Xte)
    test_acc = (test_out.argmax(1) == yte).float().mean()
    test_loss = loss_fn(test_out, yte)
print(round(train_acc.item(), 3), round(test_acc.item(), 3))
print(round(test_loss.item(), 4))

predicted = test_out.argmax(1)
for c, name in enumerate(iris.target_names):
    mask = yte == c
    print(name, int((predicted[mask] == c).sum()), "of", int(mask.sum()))

for i in (predicted != yte).nonzero().flatten().tolist():
    probs = torch.softmax(test_out[i], 0)
    print(iris.target_names[yte[i]], "called",
          iris.target_names[predicted[i]],
          [round(v, 2) for v in probs.tolist()])


# Reading the Loss Curve
torch.manual_seed(7)
model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

train_curve, test_curve = [], []
for epoch in range(301):
    loss = loss_fn(model(Xtr), ytr)
    train_curve.append(loss.item())
    with torch.no_grad():
        test_curve.append(loss_fn(model(Xte), yte).item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(round(train_curve[-1], 4), round(test_curve[-1], 4))
