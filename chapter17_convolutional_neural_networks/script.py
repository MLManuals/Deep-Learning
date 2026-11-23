"""Chapter 17: Convolutional Neural Networks -- code from the book, run top to bottom."""


# Why Dense Layers Struggle with Images
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
print(digits.images.shape, digits.images.min(), digits.images.max())

X_train, X_test, y_train, y_test = train_test_split(
    digits.images / 16.0, digits.target, test_size=0.25,
    random_state=7, stratify=digits.target)
Xtr = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
Xte = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
ytr, yte = torch.tensor(y_train), torch.tensor(y_test)
print(Xtr.shape, Xte.shape)

def shift(images, dx):
    out = torch.zeros_like(images)
    out[..., dx:] = images[..., :-dx]
    return out

digit, moved = Xte[2, 0], shift(Xte[2:3], 1)[0, 0]
print("label", yte[2].item())
print((digit != moved).sum().item(), "of 64 pixels changed")

pixels = 224 * 224 * 3
print(pixels, "inputs")
print(pixels * 1000 + 1000, "weights in one dense layer of 1000 units")
conv = nn.Conv2d(3, 64, kernel_size=3)
print(sum(p.numel() for p in conv.parameters()), "weights in 64 small filters")


# The Convolution, by Hand
image = torch.zeros(6, 6)
image[:, 3:] = 1.0
kernel = torch.tensor([[-1., 0., 1.],
                       [-1., 0., 1.],
                       [-1., 0., 1.]])

by_hand = torch.zeros(4, 4)
for row in range(4):
    for col in range(4):
        patch = image[row:row + 3, col:col + 3]
        by_hand[row, col] = (patch * kernel).sum()
for line in by_hand.tolist():
    print([int(v) for v in line])

by_torch = F.conv2d(image.view(1, 1, 6, 6), kernel.view(1, 1, 3, 3))
for line in by_torch.view(4, 4).tolist():
    print([int(v) for v in line])
print(torch.equal(by_hand, by_torch.view(4, 4)))


# Pooling and Feature Maps
fmap = torch.tensor([[1., 5., 0., 2.],
                     [3., 2., 1., 4.],
                     [0., 1., 7., 0.],
                     [2., 0., 3., 1.]])
pooled = F.max_pool2d(fmap.view(1, 1, 4, 4), 2).view(2, 2)
for line in pooled.tolist():
    print([int(v) for v in line])

torch.manual_seed(0)
cnn = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(64, 10))

h = Xtr[:5]
for layer in cnn:
    h = layer(h)
    if not isinstance(layer, nn.ReLU):
        print(type(layer).__name__, list(h.shape))


# A Small Network That Sees
loss_fn = nn.CrossEntropyLoss()

def fit(net, epochs=30, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    for epoch in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        accuracy = lambda X: (net(X).argmax(1) == yte).float().mean().item()
        return [round(accuracy(X), 3) for X in [Xte, shift(Xte, 1), shift(Xte, 2)]]

torch.manual_seed(0)
dense = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(),
                      nn.Linear(64, 10))
for name, net in [("dense", dense), ("cnn", cnn)]:
    count = sum(p.numel() for p in net.parameters())
    print(name, count, fit(net))


# Standing on Other Networks' Shoulders: a Note on Pretrained Models
low = ytr < 5
torch.manual_seed(0)
base = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten())
print(int(low.sum()), "training digits from 0 to 4")
print(list(base(Xtr[:5]).shape))

def train_on(net, X, y, epochs, params, seed=0):
    opt = torch.optim.Adam(params, lr=0.01)
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    for epoch in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()
    net.eval()

old = nn.Sequential(base, nn.Linear(64, 5))
train_on(old, Xtr[low], ytr[low], 30, old.parameters())

for p in base.parameters():
    p.requires_grad = False
torch.manual_seed(0)
new_head = nn.Linear(64, 5)
reused = nn.Sequential(base, new_head)
print(sum(p.numel() for p in reused.parameters() if p.requires_grad), "trainable")

high = torch.nonzero(ytr >= 5).flatten()
few = high[torch.randperm(len(high), generator=torch.Generator().manual_seed(7))[:50]]
X_few, y_few = Xtr[few], ytr[few] - 5
test_high = yte >= 5

train_on(reused, X_few, y_few, 100, new_head.parameters())
torch.manual_seed(0)
scratch = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(64, 5))
train_on(scratch, X_few, y_few, 100, scratch.parameters())

with torch.no_grad():
    for name, net in [("reused", reused), ("scratch", scratch)]:
        hits = net(Xte[test_high]).argmax(1) == yte[test_high] - 5
        print(name, round(hits.float().mean().item(), 3))
