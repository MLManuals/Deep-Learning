"""Chapter 20: Your First Deep Learning Project -- code from the book, run top to bottom."""


# Choosing a Problem and Data
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

digits = load_digits()
X, y = digits.images / 16.0, digits.target
X_rest, X_test, y_rest, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_rest, y_rest, test_size=0.25, random_state=7, stratify=y_rest)
print(len(X_train), len(X_val), len(X_test))

flat = lambda a: a.reshape(len(a), -1)
majority = np.bincount(y_train).argmax()
print("always guess", majority, round(np.mean(y_val == majority), 3))
logreg = LogisticRegression(max_iter=2000).fit(flat(X_train), y_train)
print("logistic regression", round(logreg.score(flat(X_val), y_val), 3))


# Designing the Network
T = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(1)
Xtr, Xva, Xte = T(X_train), T(X_val), T(X_test)
ytr, yva, yte = torch.tensor(y_train), torch.tensor(y_val), torch.tensor(y_test)
loss_fn = nn.CrossEntropyLoss()

def build(width=8, blocks=2, seed=0):
    torch.manual_seed(seed)
    layers, channels, size = [], 1, 8
    for b in range(blocks):
        out = width * 2 ** b
        layers += [nn.Conv2d(channels, out, 3, padding=1),
                   nn.BatchNorm2d(out), nn.ReLU(), nn.MaxPool2d(2)]
        channels, size = out, size // 2
    layers += [nn.Flatten(), nn.Linear(channels * size * size, 10)]
    return nn.Sequential(*layers)

def fit(net, epochs=40, batch=32, lr=0.01, seed=0, patience=None):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best, best_state, best_epoch, waited = 1e9, None, 0, 0
    for epoch in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = loss_fn(net(Xva), yva).item()
        if val < best:
            best, best_epoch, waited = val, epoch, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            waited += 1
            if patience and waited >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        accuracy = (net(Xva).argmax(1) == yva).float().mean().item()
    return round(accuracy, 3), best_epoch

torch.manual_seed(0)
dense = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(),
                      nn.Linear(64, 10))
print("dense", fit(dense)[0], sum(p.numel() for p in dense.parameters()))


# Choosing Width, Depth and Batch Size
for width in [2, 4, 8, 16, 32]:
    scores = [fit(build(width, seed=s), seed=s)[0] for s in range(3)]
    count = sum(p.numel() for p in build(width).parameters())
    print(width, count, scores, round(np.mean(scores), 3))

def average(blocks=2, **settings):
    return round(np.mean([fit(build(16, blocks, seed=s), seed=s, **settings)[0]
                          for s in range(3)]), 3)

print("one block   ", average(blocks=1))
print("two blocks  ", average(blocks=2))
print("batch 16    ", average(batch=16))
print("batch 128   ", average(batch=128))
print("rate 0.003  ", average(lr=0.003))
print("rate 0.03   ", average(lr=0.03))


# Training and Tuning Honestly
finals = []
for seed in range(5):
    net = build(16, seed=seed)
    val_acc, stopped_at = fit(net, epochs=60, seed=seed, patience=10)
    finals.append(net)
    print(seed, val_acc, stopped_at)


# Evaluating and Explaining
tests = []
for net in finals:
    with torch.no_grad():
        tests.append(round((net(Xte).argmax(1) == yte).float().mean().item(), 3))
print(tests)
print(round(np.mean(tests), 3), min(tests), max(tests))
print("logistic regression", round(logreg.score(flat(X_test), y_test), 3))

from sklearn.metrics import confusion_matrix

model = finals[0]
with torch.no_grad():
    probs = torch.softmax(model(Xte), dim=1)
predicted = probs.argmax(1)
print(confusion_matrix(yte, predicted))

wrong = torch.nonzero(predicted != yte).flatten().tolist()
for i in wrong:
    print(i, "true", yte[i].item(), "said", predicted[i].item(),
          "confidence", round(probs[i].max().item(), 3),
          "for the truth", round(probs[i, yte[i]].item(), 3))


# Presenting Your Work
torch.save({"model": finals[0].state_dict(),
            "design": {"width": 16, "blocks": 2},
            "preprocess": "pixel values divided by 16, shape (1, 8, 8)",
            "classes": list(range(10)),
            "test_accuracy_mean_of_5_seeds": round(float(np.mean(tests)), 3)},
           "digits_cnn.pt")
reloaded = build(**torch.load("digits_cnn.pt")["design"])
reloaded.load_state_dict(torch.load("digits_cnn.pt")["model"])
reloaded.eval()
with torch.no_grad():
    print(torch.equal(reloaded(Xte).argmax(1), predicted))
