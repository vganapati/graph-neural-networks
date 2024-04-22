"""
Reference: https://d2l.ai/chapter_builders-guide/custom-layer.html
"""

import torch
from torch import nn
from torch.nn import functional as F

# layer without parameters

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        return X - X.mean(axis=-1)[:,None]

layer = CenteredLayer()
print(layer(torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float32)))

net_0 = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
Y = net_0(torch.rand(4,8))
print(Y.shape)
print(Y.mean(axis=-1))

# layer with parameters

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units))
    def forward(self, X):
        linear = torch.matmul(X, self.weight) + self.bias
        return F.relu(linear)
    
linear = MyLinear(5,3)
print(linear.weight)
print(linear(torch.rand(2,5)))

net_1 = nn.Sequential(MyLinear(64,8), MyLinear(8,1))
print(net_1(torch.rand(2, 64)))

class FourierHalf(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return torch.fft.fft(X)[0:X.shape[1]//2]

print(FourierHalf()(torch.ones(5,4)))