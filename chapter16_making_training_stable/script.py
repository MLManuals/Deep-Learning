"""Chapter 16: Making Training Stable -- code from the book, run top to bottom."""


# Vanishing and Exploding Gradients
import torch
import torch.nn as nn
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=400, noise=0.08,
                    factor=0.45, random_state=7)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y)
loss_fn = nn.CrossEntropyLoss()

def deep(depth=20, width=32, act=nn.Sigmoid, init=None, bn=False, seed=0):
    torch.manual_seed(seed)
    layers, size = [], 2
    for i in range(depth):
        linear = nn.Linear(size, width)
        if init is not None:
            init(linear.weight)
            nn.init.zeros_(linear.bias)
        layers.append(linear)
        if bn:
            layers.append(nn.BatchNorm1d(width))
        layers.append(act())
        size = width
    layers.append(nn.Linear(size, 2))
    return nn.Sequential(*layers)

def gradient_sizes(net):
    net.zero_grad()
    loss_fn(net(X), y).backward()
    return [m.weight.grad.norm().item()
            for m in net if isinstance(m, nn.Linear)]

sizes = gradient_sizes(deep(act=nn.Sigmoid))
print("first layer %.1e" % sizes[0])
print("last layer  %.1e" % sizes[-1])


# Weight Initialization
def spreads(net):
    out, h = [], X
    with torch.no_grad():
        for m in net:
            h = m(h)
            if isinstance(m, nn.ReLU):
                out.append(h.std().item())
    return out

starts = {
    "tiny":    lambda w: nn.init.normal_(w, 0, 0.01),
    "default": None,
    "large":   lambda w: nn.init.normal_(w, 0, 1.0),
    "He":      lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"),
}
for name, init in starts.items():
    s = spreads(deep(act=nn.ReLU, init=init))
    print(name, ["%.2g" % s[i] for i in [0, 9, 19]])

def train(net, steps=300, lr=0.01):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for step in range(steps):
        net.train()
        opt.zero_grad()
        loss = loss_fn(net(X), y)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        accuracy = (net(X).argmax(1) == y).float().mean().item()
    return round(loss.item(), 4), round(accuracy, 3)

print("sigmoid", train(deep(act=nn.Sigmoid)))
for name, init in starts.items():
    print(name, train(deep(act=nn.ReLU, init=init)))

net = deep(act=nn.ReLU, init=starts["large"])
opt = torch.optim.SGD(net.parameters(), lr=0.01)
for step in range(3):
    opt.zero_grad()
    loss = loss_fn(net(X), y)
    loss.backward()
    opt.step()
    print("%.4g" % loss.item())


# Batch Normalization
torch.manual_seed(0)
bn = nn.BatchNorm1d(3)
batch = torch.randn(64, 3) * torch.tensor([1., 10., 100.]) \
        + torch.tensor([5., -20., 300.])
out = bn(batch)
print([round(v, 1) for v in batch.mean(0).tolist()])
print([round(v, 1) for v in batch.std(0).tolist()])
print([round(v, 2) + 0.0 for v in out.mean(0).tolist()])
print([round(v, 2) for v in out.std(0).tolist()])

print([round(v, 1) for v in bn.running_mean.tolist()])

print("sigmoid + bn", train(deep(act=nn.Sigmoid, bn=True)))
print("tiny + bn", train(deep(act=nn.ReLU, init=starts["tiny"], bn=True)))


# Putting the Training Recipe Together
class SteadyNet(nn.Module):
    def __init__(self, inputs, width, depth, outputs):
        super().__init__()
        layers, size = [], inputs
        for i in range(depth):
            linear = nn.Linear(size, width)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.BatchNorm1d(width), nn.ReLU()]
            size = width
        layers.append(nn.Linear(size, outputs))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)

print(train(SteadyNet(2, 32, 20, 2)))


# When Your Network Will Not Train: A Troubleshooting Guide
def fit_tiny(net, steps=200):
    xb, yb = X[:8], y[:8]
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn(net(xb), yb)
        loss.backward()
        opt.step()
    return round(loss.item(), 4)

print("healthy", fit_tiny(deep(act=nn.ReLU, init=starts["He"])))
print("stuck  ", fit_tiny(deep(act=nn.Sigmoid)))
