"""
https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial6/Tutorial6.ipynb#scrollTo=UqGuHXDzBTdV
"""

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import VGAE
# GAE: Graph AutoEncoder

dataset = Planetoid("tmp/", "CiteSeer", transform=T.NormalizeFeatures())
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = None

data = train_test_split_edges(data)
# data = T.RandomLinkSplit(data)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2*out_channels, cached=True) # cached due to transductive learning
        self.conv2 = GCNConv(2*out_channels, out_channels, cached=True)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2*out_channels, cached=True)
        self.conv_mu = GCNConv(2*out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2*out_channels, out_channels, cached=True)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# parameters
out_channels = 2
num_features = dataset.num_features
epochs = 300
variational = False

# model
if variational:
    model = VGAE(VariationalGCNEncoder(num_features, out_channels))
else:
    model = GAE(GCNEncoder(num_features, out_channels))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)

# initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    
    loss = model.recon_loss(z, train_pos_edge_index)
    if variational:
        loss += (1/data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)

if variational:
    writer = SummaryWriter('runs/VGAE_experiment_2d_100_epochs')
else:
    writer = SummaryWriter('runs/GAE1_experiment_2d_100_epochs')

for epoch in range(epochs):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    writer.add_scalar('auc_train', auc, epoch)
    writer.add_scalar('ap train', ap, epoch)

# tensorboard --logdir runs