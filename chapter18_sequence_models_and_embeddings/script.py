"""Chapter 18: Sequence Models and Embeddings -- code from the book, run top to bottom."""


# Data That Comes in Order
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def make_order(n, length, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(3, 10, (n, length), generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    for i in range(n):
        spots = torch.randperm(length, generator=g)[:2].sort().values
        first, second = (1, 2) if y[i] == 1 else (2, 1)
        X[i, spots[0]], X[i, spots[1]] = first, second
    return X, y

X, y = make_order(3, 12, seed=5)
for row, label in zip(X.tolist(), y.tolist()):
    print(row, label)


# Embeddings, Turning Symbols into Vectors
torch.manual_seed(0)
emb = nn.Embedding(10, 16)
print(emb.weight.shape)
looked_up = emb(torch.tensor([4]))
one_hot = torch.zeros(1, 10)
one_hot[0, 4] = 1.0
print(torch.equal(looked_up, one_hot @ emb.weight))


# The Recurrent Idea
torch.manual_seed(0)
rnn = nn.RNN(input_size=4, hidden_size=3, batch_first=True)
inputs = torch.randn(1, 5, 4)

h = torch.zeros(3)
for t in range(5):
    h = torch.tanh(rnn.weight_ih_l0 @ inputs[0, t] + rnn.bias_ih_l0
                   + rnn.weight_hh_l0 @ h + rnn.bias_hh_l0)
outputs, last = rnn(inputs)
print([round(v, 4) for v in h.tolist()])
print([round(v, 4) for v in last[0, 0].tolist()])
print(torch.allclose(h, last[0, 0]))

class Bag(nn.Module):
    def __init__(self, vocab=10, dim=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, 2)
    def forward(self, x):
        return self.out(self.emb(x).mean(1))

class Reader(nn.Module):
    def __init__(self, cell, vocab=10, dim=16, hidden=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.rnn = cell(dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 2)
    def forward(self, x):
        states, _ = self.rnn(self.emb(x))
        return self.out(states[:, -1])

loss_fn = nn.CrossEntropyLoss()

def run(model, maker, length, epochs=20, seed=0):
    Xtr, ytr = maker(2000, length, 1)
    Xte, yte = maker(500, length, 2)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return round((model(Xte).argmax(1) == yte).float().mean().item(), 3)

torch.manual_seed(0)
bag = Bag()
torch.manual_seed(0)
reader = Reader(nn.RNN)
print("bag", run(bag, make_order, 30))
print("rnn", run(reader, make_order, 30))

def make_first(n, length, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(3, 10, (n, length), generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    X[:, 0] = torch.where(y == 1, 1, 2)
    return X, y

X, y = make_first(64, 100, seed=3)
for cell in [nn.RNN, nn.LSTM]:
    torch.manual_seed(0)
    model = Reader(cell)
    e = model.emb(X).detach().requires_grad_(True)
    states, _ = model.rnn(e)
    loss_fn(model.out(states[:, -1]), y).backward()
    size = e.grad.norm(dim=(0, 2))
    print(cell.__name__, ["%.0e" % size[99 - back].item()
                          for back in [0, 10, 25, 50, 99]])

def open_forget_gate(model):
    size = model.rnn.hidden_size
    for name, p in model.rnn.named_parameters():
        if "bias" in name:
            with torch.no_grad():
                p[size:2 * size].fill_(1.0)
    return model

for seed in [0, 1, 2]:
    results = []
    for build in [lambda: Reader(nn.RNN), lambda: Reader(nn.LSTM),
                  lambda: open_forget_gate(Reader(nn.LSTM))]:
        torch.manual_seed(seed)
        results.append(run(build(), make_first, 50, epochs=30, seed=seed))
    print(seed, results)
