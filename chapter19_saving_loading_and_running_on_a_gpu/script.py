"""Chapter 19: Saving, Loading and Running on a GPU -- code from the book, run top to bottom."""


# Checkpoints, Saving a Trained Model
import os
import torch
import torch.nn as nn
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
loss_fn = nn.CrossEntropyLoss()

def make_net(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))

def train_steps(net, opt, n):
    for step in range(n):
        opt.zero_grad()
        loss = loss_fn(net(Xtr), ytr)
        loss.backward()
        opt.step()
    return loss.item()

net = make_net()
opt = torch.optim.Adam(net.parameters(), lr=0.01)
print(round(train_steps(net, opt, 100), 6))

print(list(net.state_dict().keys()))
torch.save(net.state_dict(), "iris_weights.pt")
print(os.path.getsize("iris_weights.pt"), "bytes")

net = make_net()
opt = torch.optim.Adam(net.parameters(), lr=0.01)
train_steps(net, opt, 50)
torch.save({"step": 50,
            "model": net.state_dict(),
            "optimizer": opt.state_dict(),
            "mean": mean.tolist(),
            "std": std.tolist()}, "iris_checkpoint.pt")
print(os.path.getsize("iris_checkpoint.pt"), "bytes")

checkpoint = torch.load("iris_checkpoint.pt")

resumed = make_net(seed=99)
resumed_opt = torch.optim.Adam(resumed.parameters(), lr=0.01)
resumed.load_state_dict(checkpoint["model"])
resumed_opt.load_state_dict(checkpoint["optimizer"])
print("resumed", round(train_steps(resumed, resumed_opt, 50), 6))

forgetful = make_net(seed=99)
forgetful_opt = torch.optim.Adam(forgetful.parameters(), lr=0.01)
forgetful.load_state_dict(checkpoint["model"])
print("weights only", round(train_steps(forgetful, forgetful_opt, 50), 6))


# Loading for Inference
model = make_net(seed=123)
model.load_state_dict(checkpoint["model"])
model.eval()

new_flower = [5.9, 3.0, 5.1, 1.8]
scaled = (torch.tensor(new_flower) - torch.tensor(checkpoint["mean"])) \
         / torch.tensor(checkpoint["std"])
with torch.no_grad():
    probs = torch.softmax(model(scaled.float().unsqueeze(0)), dim=1)[0]
print([round(p, 3) for p in probs.tolist()])
print(iris.target_names[probs.argmax().item()])


# Moving to the GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)

model = make_net().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
Xtr_d, ytr_d = Xtr.to(device), ytr.to(device)
for step in range(100):
    opt.zero_grad()
    loss = loss_fn(model(Xtr_d), ytr_d)
    loss.backward()
    opt.step()
print(next(model.parameters()).device.type, round(loss.item(), 4))


# Reproducibility and Seeds
a, b, c = make_net(seed=0), make_net(seed=0), make_net(seed=1)
print(torch.equal(a[0].weight, b[0].weight))
print(torch.equal(a[0].weight, c[0].weight))

scores = []
for seed in range(10):
    net = make_net(seed)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    train_steps(net, opt, 100)
    with torch.no_grad():
        scores.append(round((net(Xte).argmax(1) == yte).float().mean().item(), 3))
print(scores)
print(round(sum(scores) / len(scores), 3), min(scores), max(scores))
