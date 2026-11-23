"""Chapter 10: Building a Network with nn.Module -- code from the book, run top to bottom."""


# Layers the PyTorch Way
import torch
import torch.nn as nn

torch.manual_seed(7)
layer = nn.Linear(2, 3)
print(layer)
print(layer.weight.shape, layer.bias.shape)

print(layer.weight.data.round(decimals=3))
print(layer.weight.requires_grad)

student = torch.tensor([8.0, 3.0])
with torch.no_grad():
    print(layer(student).round(decimals=3))


# Writing an nn.Module
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        h = torch.relu(self.hidden(x))
        return self.output(h)

torch.manual_seed(7)
net = TinyNet()
print(net)


# Putting Layers Together
torch.manual_seed(7)
deep = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 3),
    nn.ReLU(),
    nn.Linear(3, 1))
print(deep)


# Counting the Parameters
print(sum(p.numel() for p in net.parameters()))

for name, p in net.named_parameters():
    print(name, tuple(p.shape), p.numel())

print(sum(p.numel() for p in deep.parameters()))

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
print(type(optimizer).__name__, optimizer.param_groups[0]["lr"])
