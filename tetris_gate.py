"""
Reference: https://github.com/e3nn/e3nn/blob/main/examples/tetris_gate.py
"""
import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter

from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.math import soft_one_hot_linspace

from tetris_polynomial import tetris

class Convolution(torch.nn.Module):
    def __init__(self, irreps_in, irreps_sh, irreps_out, num_neighbors):
        super().__init__()

        self.num_neighbors = num_neighbors

        tp = FullyConnectedTensorProduct(irreps_in1=irreps_in, irreps_in2=irreps_sh, irreps_out=irreps_out, internal_weights=False, shared_weights=False)
        self.fc = FullyConnectedNet([3, 256, tp.weight_numel], torch.relu)
        self.tp = tp
        self.irreps_out = self.tp.irreps_out
    
    def forward(self, node_features, edge_src, edge_dst, edge_attr, edge_scalars):
        weight = self.fc(edge_scalars)
        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        node_features = scatter(edge_features, edge_dst, dim=0).div(self.num_neighbors**0.5)
        return node_features

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_neighbors = 3.8 # typical number of neighbors
        self.irreps_sh = o3.Irreps.spherical_harmonics(3, p=1)

        irreps = self.irreps_sh

        # First layer with gate
        gate = Gate("16x0e + 16x0o", [torch.relu, torch.abs], "8x0e + 8x0o + 8x0e + 8x0o", [torch.relu, torch.tanh, torch.relu, torch.tanh], "16x1o + 16x1e")
        self.conv = Convolution(irreps, self.irreps_sh, gate.irreps_in, self.num_neighbors)
        self.gate = gate
        irreps = self.gate.irreps_out

        # Final layer
        self.final = Convolution(irreps, self.irreps_sh, "8x0e", self.num_neighbors)
        self.irreps_out = self.final.irreps_out
    
    def forward(self, data):
        
        num_nodes = 4 # typical number of nodes
        edge_src, edge_dst = radius_graph(x=data.pos, r=2.5, batch=data.batch)
        edge_vec = data.pos[edge_src] - data.pos[edge_dst]
        edge_attr = o3.spherical_harmonics(l=self.irreps_sh, x=edge_vec, normalize=True, normalization="component")
        edge_length_embedded = (soft_one_hot_linspace(x=edge_vec.norm(dim=1), start=0.5, end=2.5, number=3, basis="smooth_finite", cutoff=True) * 3**0.5)

        x = scatter(edge_attr, edge_dst, dim=0).div(self.num_neighbors**0.5) # initial features
        x = self.conv(x, edge_src, edge_dst, edge_attr, edge_length_embedded)
        x = self.gate(x)
        x = self.final(x, edge_src, edge_dst, edge_attr, edge_length_embedded)

        return scatter(x, data.batch, dim=0).div(num_nodes**0.5)

def main():
    data, labels = tetris()
    f = Network()

    print("Build a model:")
    print(f)

    optim = torch.optim.Adam(f.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for step in range(300):
        pred = f(data)
        loss = loss_fn(pred, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 10 == 0:
            accuracy = (torch.argmax(pred, dim=1) == torch.argmax(labels, dim=1)).sum().item()/len(labels)
            print(f"epoch {step:5d} | loss {loss:<10.5f} | {100 * accuracy:5.1f}% accuracy")
        
    # Check equivariance
    print("Testing equivariance directly...")
    rotated_data, _ = tetris()
    error = f(rotated_data) - f(data)
    error = error.abs().max().item()
    print(f"Equivariance error = {error:.1e}")
    assert error < 1e-4

if __name__ == "__main__":
    main()
