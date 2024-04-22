import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Simple PyTorch Implementation of the Graph Attention Layer
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # Xavier initialization of weights
        self.W = nn.Parameter(torch.zeros(size=(in_features,out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2*out_features,1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # Leaky ReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        
        # Linear transformation
        h = torch.matmul(input, self.W)
        N = input.shape[0] # number of nodes

        adj = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]), torch.Size([N,N])).to_dense()

        # Attention mechanism
        a_input = torch.cat([h.repeat(1,N).view(N*N, -1), h.repeat(N,1)], dim=1).view(N, -1, 2*self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # Masked attention (mask out nodes that are not adjacent)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj>0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention,h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import matplotlib.pyplot as plt

name_data = 'Cora'
dataset = Planetoid(root='/tmp', name = name_data)
dataset.transform = T.NormalizeFeatures()

class GAT(torch.nn.Module):
    def __init__(self, slow=False):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1

        if slow:
            self.conv1 = GATLayer(dataset.num_features, self.hid*self.in_head, dropout=0.6, alpha=0.2, concat=True)
            self.conv2 = GATLayer(self.hid*self.in_head, dataset.num_classes*self.out_head, dropout=0.6, alpha=0.2, concat=False)
        else:
            self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout=0.6)
            self.conv2 = GATConv(self.hid*self.in_head, dataset.num_classes, concat=False,
                                heads=self.out_head, dropout=0.6)
        


    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x,dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT().to(device)
data = dataset[0].to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

model.train()
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    if epoch%200 == 0:
        print(loss)
    
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct/data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))