import torch
from torch.nn import Linear, Parameter, Sequential, ReLU
from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data

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

        # add self_loops to adjancency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # linearly transform node feature matrix
        x = self.lin(x)

        # compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # start propagating messages
        out = self.propagate(edge_index, x=x, norm=norm)

        # apply final bias vector
        out = out + self.bias

        return out
    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        return norm[:,None] * x_j

    def update(self, x_aggr):
        return x_aggr


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')
        self.mlp = Sequential(Linear(2 * in_channels, out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels))
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        # x_i has the shape [E, in_channels]
        # x_j has the shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1) # tmp has shape [E, 2*in_channels]
        return self.mlp(tmp)

class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6):
        super().__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, edge_index, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super().forward(x, edge_index)

if __name__ == '__main__':
    num_nodes = 3
    num_features_in = 16
    num_features_out = 32
    edge_index = torch.tensor([[0,1,1,2],
                               [1,0,2,1]], dtype=torch.long)
    x = torch.randn([num_nodes, num_features_in])
    conv = GCNConv(num_features_in, num_features_out)
    print(x.shape)
    x = conv(x, edge_index)
    print(x.shape)

    num_nodes = 40
    num_features_in = 3
    num_features_out = 128
    x = torch.randn([num_nodes, num_features_in])
    conv = DynamicEdgeConv(3, 128, k=6)
    print(x.shape)
    x = conv(x, edge_index)
    print(x.shape)