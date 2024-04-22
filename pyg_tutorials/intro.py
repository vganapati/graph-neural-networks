'''
https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
'''

import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],[1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

# Transfer data object to GPU
device = torch.device('cuda')
data = data.to(device)

# ENZYMES dataset

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
dataset_enzymes = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
dataset_enzymes = dataset_enzymes.shuffle()
loader_enzymes = DataLoader(dataset_enzymes, batch_size=32, shuffle=False)

for batch in loader_enzymes:
    print(batch)
    print(batch.num_graphs)

# ENZYMES dataset with node_attr
    
from torch_geometric.utils import scatter
dataset_enzymes_1 = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset_enzymes_1, batch_size=32, shuffle=True)

for data in loader:
    print(data)
    print(data.num_graphs)
    x = scatter(data.x, data.batch, dim=0, reduce='mean')
    print(x.shape)

# Cora dataset
from torch_geometric.datasets import Planetoid
dataset_cora = Planetoid(root='/tmp/Cora', name='Cora')

# ShapeNet dataset
from torch_geometric.datasets import ShapeNet
dataset_shapenet = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])

print(dataset_shapenet[0])

## Convert point clouds to graphs by getting nearest neighbors

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

dataset_graphs = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'], pre_transform=T.KNNGraph(k=6), force_reload=True)
print(dataset_graphs[0])



