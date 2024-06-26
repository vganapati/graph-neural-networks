"""
https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
"""

import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix
        x = self.lin(x)

        # Step 3: Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype = x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector
        out = out + self.bias

        return out
    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features
        return norm.view(-1, 1) * x_j

class GraphEquivariantLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        
        self.reset_parameters()
    
    def resest_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()
    
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2,e]

        row, col = edge_index
        
        deg = degree(col, x.size(0), dtype = x.dtype)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv
        out = self.propagate(edge_index, x=x)
        x_new = out[:,0:3]
        features = out[:,3:]
        x_new = norm.view(-1,1)*x_new
        x_new = x + x_new
        return x_new, features

    def message(self, x_i, x_j):
        edge_distances_sqr = torch.sum((x_i - x_j)**2, axis=1)
        x_new = (self.lin(edge_distances_sqr[:,None])+self.bias)*(x_i - x_j)
        features = (self.lin(edge_distances_sqr[:,None])+self.bias) #edge_distances_sqr[:,None] #(self.lin(edge_distances_sqr[:,None])+self.bias) #torch.sum(x_new**2, axis=1)[:,None]
        out = torch.concatenate((x_new, features), axis=1)
        # print(features)
        return out