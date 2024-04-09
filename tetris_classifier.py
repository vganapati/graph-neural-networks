"""
Conda environment setup:

conda create --name pyg python=3.10
conda activate pyg
pip3 install torch torchvision torchaudio
pip install torch_geometric
"""


"""
Shape 0:
* -- *
|    |
* -- *

Shape 1:
     *
     |
* -- * -- *

Shape 2:
*
|
*
|
*
|
*

Shape 3:
(assume flips are the same shape)
* -- *
|
*
|
*

Shape 4:
(assume flips are the same shape)
     *
     |
* -- *
|
*

GCN to classify/transform vs. E(3) network
"""

import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool

# Create training set

# edge_index assumes every point is connected
edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 1, 2, 3, 2, 3, 3],
                           [1, 2, 3, 2, 3, 3, 0, 0, 0, 1, 1, 2]], dtype=torch.long)

edge_index_0 = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 0],
                            [1, 2, 3, 0, 0, 1, 2, 3,]], dtype=torch.long)

edge_index_1 = torch.tensor([[0, 1, 1, 1, 3, 2],
                            [1, 3, 2, 0, 1, 1]], dtype=torch.long)

edge_index_2 = torch.tensor([[0, 1, 2, 1, 2, 3],
                            [1, 2, 3, 0, 1, 2]], dtype=torch.long)

edge_index_3 = torch.tensor([[0, 1, 2, 1, 2, 3],
                            [1, 2, 3, 0, 1, 2]], dtype=torch.long)

edge_index_4 = torch.tensor([[0, 1, 2, 1, 2, 3],
                            [1, 2, 3, 0, 1, 2]], dtype=torch.long)

node_attributes_0 = torch.tensor([[0, 0, 0],
                                  [1, 0, 0],
                                  [1, 1, 0],
                                  [0, 1, 0]], dtype=torch.float)

node_attributes_1 = torch.tensor([[-1, 0, 0],
                                  [0, 0, 0],
                                  [1, 0, 0],
                                  [0, 1, 0]], dtype=torch.float)

node_attributes_2 = torch.tensor([[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 2, 0],
                                  [0, 3, 0]], dtype=torch.float)

node_attributes_3 = torch.tensor([[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 2, 0],
                                  [1, 2, 0]], dtype=torch.float)

node_attributes_4 = torch.tensor([[0, 0, 0],
                                  [0, 1, 0],
                                  [1, 1, 0],
                                  [1, 2, 0]], dtype=torch.float)
num_classes = 5
num_features = 3
batch_size = 5

data_0 = Data(x=node_attributes_0, edge_index=edge_index, y=torch.tensor([0]))
data_1 = Data(x=node_attributes_1, edge_index=edge_index, y=torch.tensor([1]))
data_2 = Data(x=node_attributes_2, edge_index=edge_index, y=torch.tensor([2]))
data_3 = Data(x=node_attributes_3, edge_index=edge_index, y=torch.tensor([3]))
data_4 = Data(x=node_attributes_4, edge_index=edge_index, y=torch.tensor([4]))

data_list = [data_0, data_1, data_2, data_3, data_4]
data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)


# Transformation

# Random transformation by 3D rotation and translation in a 5x5x5 box

# 3D rotation, uniform on unit sphere: https://paulbourke.net/geometry/randomvector/

# u is uniformly randomly distributed from 0 to 1
u = torch.rand(1)

theta_x = torch.acos(2*u - 1)
theta_z = 2*np.pi*u

# perform x rotation first
rot_x_mat = torch.tensor([[1, 0, 0],
                          [1, torch.cos(theta_x), -torch.sin(theta_x)],
                          [0, torch.sin(theta_x), torch.cos(theta_x)]])

rot_z_mat = torch.tensor([[torch.cos(theta_z), -torch.sin(theta_z), 0],
                          [torch.sin(theta_z), torch.cos(theta_z), 0],
                          [0, 0, 1]])

# translation vector
translate_x = 5*torch.rand(1)
translate_y = 5*torch.rand(1)
translate_z = 5*torch.rand(1)

translate_vec = torch.tensor([[translate_x],
                              [translate_y],
                              [translate_z]])


# Classification

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
    def forward(self, x, edge_index, batch, dropout_prob=0):
        # obtain node embeddings
        x = self.conv1(x, edge_index)
        # readout layer
        x = global_add_pool(x, batch) # output shape is [batch_size, hidden_channels]
        # final classifier
        x = F.dropout(x, p=dropout_prob, training=self.training)
        x = self.lin(x)
        return x

model = GCN(hidden_channels=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    total_loss = 0
    for data in data_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        total_loss += loss
        loss.backward()
        optimizer.step()
    return total_loss

def test(transform=True):
    model.eval()
    correct = 0
    for data in data_loader:
        # XXX transform x features
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(data_loader.dataset)

num_epochs = 100

for epoch in range(num_epochs):
    total_loss = train()
    # i=0
    # for name, param in model.state_dict().items():
    #     if i==1:
    #         print(name, param)
    #         break
    #     i += 1
    train_acc = test(transform=False)
    test_acc = test(transform=True)
    print(f'Epoch: {epoch:03d}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')