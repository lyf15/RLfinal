import torch, torch.nn as nn, torch.nn.functional as F

def mlp(in_dim, out_dim, hidden_dim=64, n=2, activation="tanh"):
    if activation == "tanh": act = nn.Tanh 
    layers = []
    las = in_dim
    for _ in range(n):
        layers += [nn.Linear(las, hidden_dim), act()]
        las = hidden_dim
    layers += [nn.Linear(las, out_dim)]
    return nn.Sequential(*layers)

class MLP1(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, n=2, activation="tanh"):
        super().__init__()
        self.mlp = mlp(in_dim, out_dim, hidden_dim, n, activation)
        self.log_std = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        mean = self.mlp(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std), mean, std

class MLP2(nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden_dim=64, n=2, activation="tanh"):
        super().__init__()
        self.mlp = mlp(in_dim, out_dim, hidden_dim, n, activation)
    def forward(self,x):
        return self.mlp(x).squeeze(-1)