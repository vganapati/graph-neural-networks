"""
Convert a tetris block into another one with e3nn
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter
import torch.nn.functional as F

from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.math import soft_one_hot_linspace

def tetris(rotate=True, test_inversion=False):
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


    pos_labels = [
        [(0,0,0),(1,0,0),(1,1,0),(2,1,0)], # zigzag
        [(0,0,0),(0,0,1),(1,0,0),(1,1,0)], # chiral_shape_1
        [(0,0,0),(0,0,1),(1,0,0),(1,-1,0)], # chiral_shape_2
        [(0,0,0),(1,0,0),(0,1,0),(1,1,0)], # square
        [(0,0,0),(0,0,1),(0,0,2),(0,0,3)], # line
        [(0,0,0),(0,0,1),(0,1,0),(1,0,0)], # corner
        [(0,0,0),(0,0,1),(0,0,2),(0,1,0)], # L
        [(0,0,0),(0,0,1),(0,0,2),(0,1,1)], # T
    ]

    pos_labels = torch.tensor(pos_labels, dtype=torch.get_default_dtype())

    if rotate:
        # apply random rotation
        rot = o3.rand_matrix(len(pos))
        if test_inversion:
            rot = -1*rot
        pos = torch.einsum("zij,zaj->zai", rot, pos)
        pos_labels = torch.einsum("zij,zaj->zai", rot, pos_labels)
    else:
        rot = torch.tile(torch.eye(3)[None],dims=(len(pos),1,1))

    dataset = [Data(pos=p) for p in pos]
    data = next(iter(DataLoader(dataset, batch_size=len(dataset))))

    dataset_labels = [Data(pos=p) for p in pos_labels]
    data_labels = next(iter(DataLoader(dataset_labels, batch_size=len(dataset))))
    return data, data_labels, rot

class Convolution(torch.nn.Module):
    def __init__(self, irreps_in, irreps_sh, irreps_out, num_neighbors, num_shapes):
        super().__init__()
        self.num_neighbors = num_neighbors
        tp = FullyConnectedTensorProduct(irreps_in1=irreps_in, irreps_in2=irreps_sh, 
                                         irreps_out=irreps_out, internal_weights=False, 
                                         shared_weights=False)
        self.fc = FullyConnectedNet([3, 256, tp.weight_numel], torch.relu)
        self.tp = tp
        self.irreps_out = self.tp.irreps_out
        self.num_shapes = num_shapes

    def forward(self, node_features, edge_src, edge_dst, edge_attr, edge_scalars):
        weight = self.fc(edge_scalars)
        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        node_features = scatter(edge_features, edge_dst, dim=0, dim_size=4*self.num_shapes).div(self.num_neighbors**0.5)
        return node_features

class UpdatePositionLayer(torch.nn.Module):
    def __init__(self, parity, num_shapes):
        super().__init__()
        self.num_neighbors = 3.8
        self.irreps_sh = o3.Irreps.spherical_harmonics(3, p=parity)
        irreps = self.irreps_sh
        
        # First layer with gate
        gate = Gate("16x0e + 16x0o", [torch.relu, torch.abs],
                    "8x0e + 8x0o + 8x0e + 8x0o",
                    [F.silu, torch.tanh, F.silu, torch.tanh],
                    "16x1o + 16x1e")
        self.conv = Convolution(irreps, self.irreps_sh, gate.irreps_in, self.num_neighbors, num_shapes)
        self.gate = gate
        irreps = self.gate.irreps_out

        # Final layer
        if parity == 1:
            final_irreps_pos = "1x1e"
        else:
            final_irreps_pos = "1x1o"

        self.final_pos_layer = Convolution(irreps, self.irreps_sh, final_irreps_pos, self.num_neighbors, num_shapes)
        self.final_x_layer = Convolution(irreps, self.irreps_sh, self.irreps_sh, self.num_neighbors, num_shapes)
    
    def forward(self, data, position, init_layer=True, x=None, rot=None):
        num_nodes = 4 # typical number of nodes
        edge_src, edge_dst = radius_graph(x=position, r=2.5, batch=data.batch)
        edge_vec = position[edge_src] - position[edge_dst]
        edge_attr = o3.spherical_harmonics(l=self.irreps_sh, x=edge_vec, normalize="True",
                                           normalization="component")
        edge_length_embedded = soft_one_hot_linspace(x=edge_vec.norm(dim=1), start=0.5, end=2.5, number=3, basis="smooth_finite", cutoff=True) * 3**0.5
        if init_layer:
            rot = rot[data.batch]
            x = self.irreps_sh.D_from_matrix(rot)@torch.ones([edge_attr.shape[1],1]) # initial features
            x = x.squeeze(-1)
        x = self.conv(x, edge_src, edge_dst, edge_attr, edge_length_embedded)
        x = self.gate(x)
        x_pos = self.final_pos_layer(x,edge_src, edge_dst, edge_attr, edge_length_embedded)
        x = self.final_x_layer(x,edge_src, edge_dst, edge_attr, edge_length_embedded)
        return x, x_pos

class Network(torch.nn.Module):
    def __init__(self, parity, num_shapes):
        super().__init__()
        self.layer_0 = UpdatePositionLayer(parity, num_shapes)
        self.layer_1 = UpdatePositionLayer(parity, num_shapes)
        self.layer_2 = UpdatePositionLayer(parity, num_shapes)

    def forward(self, data, rot):
        position = data.pos
        # add small initial values to the tensors

        x, x_pos = self.layer_0(data, position, init_layer=True, rot=rot)
        position = position + x_pos
    
        x, x_pos = self.layer_1(data, position, init_layer=False, x=x)
        position = position + x_pos

        x, x_pos = self.layer_2(data, position, init_layer=False, x=x)
        position = position + x_pos

        # x = self.layer_2(data, position)
        # position = position + x

        return position

def main():
    data, data_labels, rot = tetris(rotate=False)
    num_shapes = len(data.pos)//4
    parity = 1 # if parity == -1, cannot distinguish chiral shapes
    if parity == -1:
        test_inversion = True
    else:
        test_inversion = False
    f = Network(parity, num_shapes)

    print("Build a model:")
    print(f)

    optim = torch.optim.Adam(f.parameters(), lr=1e-3)

    for step in range(400):
        pred = f(data, rot)
        loss = (pred - data_labels.pos).pow(2).sum() # XXX fix loss function to first match points then calculate MSE

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 10 == 0:
            print(f"epoch {step:5d} | loss {loss:<10.5f}")
    
    # Check equivariance
    print("Testing equivariance directly...")
    for batch in range(torch.max(data.batch)+1):
        rotated_data, rotated_data_labels, rand_mat = tetris(rotate=True, test_inversion=test_inversion)
        pred_0 = (rand_mat[batch] @ (f(data, rot)[data.batch == batch]).T).T
        pred_1 = f(rotated_data, rand_mat)[rotated_data.batch == batch]
        print(torch.max(torch.abs(pred_0-pred_1)))
        assert torch.max(torch.abs(pred_0-pred_1))<1e-3
        
        shape_actual = rotated_data_labels.pos[rotated_data.batch == batch]
        shape_pred = pred_1
        plot_shape(shape_actual.detach().numpy(), shape_pred.detach().numpy(), batch)

    breakpoint()

    # # Make plots
    # initial_shape = rotated_data
    # final_shape = f(rotated_data, rand_mat)
    # desired_shape = rotated_data_labels
    # for batch in range(torch.max(initial_shape.batch)+1):
    #     shape_actual = desired_shape.pos[initial_shape.batch == batch]
    #     shape_pred = final_shape[initial_shape.batch == batch]
    #     plot_shape(shape_actual.detach().numpy(), shape_pred.detach().numpy(), batch)
    breakpoint()

def plot_shape(shape_actual, shape_pred, batch):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(shape_actual[:,0], shape_actual[:,1], shape_actual[:,2], c="r", s=50, marker="s")
    ax.scatter(shape_pred[:,0], shape_pred[:,1], shape_pred[:,2], c="b", s=100, marker="o")
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-2, 2])
    plt.savefig("tetris_block" + str(batch) + ".png")

if __name__ == '__main__':
    main()
