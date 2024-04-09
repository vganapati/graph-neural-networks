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
from torch_geometric.nn import GCNConv, global_mean_pool

# Create training set

# assume every point is connected, same for every shape
edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 1, 2, 3, 2, 3, 3],
                           [1, 2, 3, 2, 3, 3, 0, 0, 0, 1, 1, 2]], dtype=torch.long)
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
batch_size = 2
labels = F.one_hot(torch.arange(0,num_classes))
data_0 = Data(x=node_attributes_0, edge_index=edge_index, y=torch.tensor([0]))
data_1 = Data(x=node_attributes_0, edge_index=edge_index, y=labels[1])
data_2 = Data(x=node_attributes_0, edge_index=edge_index, y=labels[2])
data_3 = Data(x=node_attributes_0, edge_index=edge_index, y=labels[3])
data_4 = Data(x=node_attributes_0, edge_index=edge_index, y=labels[4])

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

breakpoint()
# Classification

class GCN(torch.nn.Module):
     def __init__(self, hidden_channels):
          super(GCN, self).__init__()
          self.conv1 = GCNConv(num_features, hidden_channels)
          self.conv2 = GCNConv(hidden_channels, hidden_channels)
          self.lin = Linear(hidden_channels, num_classes)
     def forward(self, x, edge_index, batch):
          # obtain node embeddings
          x = self.conv1(x, edge_index)
          x = x.relu()
          x = self.conv2(x, edge_index)

          # readout layer
          x = global_mean_pool(x, batch) # output shape is [batch_size, hidden_channels]

          # final classifier
          x = F.dropout(x, p=0.5, training=self.training)
          x = self.lin(x)
          return x

model = GCN(hidden_channels=6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
     model.train()

     for data in data_loader:
          optimizer.zero_grad()
          out = model(data.x, data.edge_index, data.batch)
          loss = criterion(out, data.y)
          loss.backward()
          optimizer.step()

def test(transform=True):
     model.eval()
     correct = 0
     for data in data_loader:
          out = model(data.x, data.edge_index, data.batch)
          pred = out.argmax