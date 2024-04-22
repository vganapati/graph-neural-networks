"""
Reference: https://github.com/e3nn/e3nn/blob/main/examples/tetris_polynomial.py
"""

import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter

from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct

def tetris():
    pos = [
        [(0,0,0),(0,0,1),(1,0,0),(1,1,0)], # chiral_shape_1
        [(0,0,0),(0,0,1),(1,0,0),(1,-1,0)], # chiral_shape_2
        [(0,0,0),(1,0,0),(0,1,0),(1,1,0)], # square
        [(0,0,0),(0,0,1),(0,0,2),(0,0,3)], # line
        [(0,0,0),(0,0,1),(0,1,0),(1,0,0)], # corner
        [(0,0,0),(0,0,1),(0,0,2),(0,1,0)], # L
        [(0,0,0),(0,0,1),(0,0,2),(0,1,1)], # T
        [(0,0,0),(1,0,0),(1,1,0),(2,1,0)], # zigzag
    ]
    
    pos = torch.tensor(pos, dtype=torch.get_default_dtype())

    labels = torch.tensor(
        [
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1],
        ],
        dtype=torch.get_default_dtype(),
    )

    # apply random rotation
    pos = torch.einsum("zij,zaj->zai", o3.rand_matrix(len(pos)), pos)

    dataset = [Data(pos=p) for p in pos]
    data = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    return data, labels

class InvariantPolynomial(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.irreps_sh = o3.Irreps.spherical_harmonics(3,p=1)
        irreps_mid = o3.Irreps("64x0e + 24x1e + 24x1o + 16x2e + 16x2o")
        irreps_out = o3.Irreps("8x0e")

        self.tp1 = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_sh,
            irreps_in2=self.irreps_sh,
            irreps_out=irreps_mid
        )

        self.tp2 = FullyConnectedTensorProduct(
            irreps_in1=irreps_mid,
            irreps_in2=self.irreps_sh,
            irreps_out=irreps_out,
        )
        self.irreps_out = self.tp2.irreps_out
    
    def forward(self, data):
        num_neighbors = 2
        num_nodes = 4
        edge_src, edge_dst = radius_graph(x=data.pos, r=1.1, batch=data.batch)
        edge_vec = data.pos[edge_src] - data.pos[edge_dst]
        edge_sh = o3.spherical_harmonics(l=self.irreps_sh, x=edge_vec, normalize=False, normalization="component",)

        # create initial features
        node_features = scatter(edge_sh, edge_dst, dim=0).div(num_neighbors**0.5)

        edge_features = self.tp1(node_features[edge_src], edge_sh)
        node_features = scatter(edge_features, edge_dst, dim=0).div(num_neighbors**0.5)

        edge_features = self.tp2(node_features[edge_src], edge_sh)
        node_features = scatter(edge_features, edge_dst, dim=0).div(num_neighbors**0.5)
        # sum all node features per graph
        return scatter(node_features, data.batch, dim=0).div(num_nodes**0.5)

def main():
    data, labels = tetris()
    f = InvariantPolynomial()

    optim = torch.optim.Adam(f.parameters(), lr=1e-2)
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

    print("Testing equivariance directly...")
    rotated_data, _ = tetris()
    error = f(rotated_data) - f(data)
    error = error.abs().max().item()
    print(f"Equivariance error = {error:.1e}")
    assert error < 1e-5

    print("Testing equivariance directly...")
    rotated_data, _ = tetris()
    error = f(rotated_data) - f(data)
    error = error.abs().max().item()
    print(f"Equivariance error = {error:.1e}")
    assert error < 1e-5

if __name__ == "__main__":
    main()
