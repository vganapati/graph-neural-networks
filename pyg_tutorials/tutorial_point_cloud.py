"""
Point Cloud Processing
Reference: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/point_cloud.html
"""

import torch
from torch_geometric.datasets import GeometricShapes
import torch_geometric.transforms as T
from torch_geometric.nn import global_max_pool, MessagePassing
from torch_geometric.loader import DataLoader

train_dataset = GeometricShapes(root='data/GeometricShapes', train=True)
train_dataset.transform = T.Compose([T.SamplePoints(num=256), T.KNNGraph(k=6)])

test_dataset = GeometricShapes(root='data/GeometricShapes', train=False)
test_dataset.transform = T.Compose([T.SamplePoints(num=256), T.KNNGraph(k=6)])

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10)


class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')

        # MLP
        # number of input features includes point dimensionality
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels + 3, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels),
        )
    def forward(self, h, pos, edge_index):
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self, h_j, pos_j, pos_i):
        # h_j is the features of the neighbors [num_edges, in_channels]
        # pos_j is the position of neighbors [num_edges, 3]
        # pos_i the central node position [num_edges, 3]
        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)

class PointNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = PointNetLayer(3,32)
        self.conv2 = PointNetLayer(32,32)
        self.classifier = torch.nn.Linear(32, train_dataset.num_classes)
    
    def forward(self, pos, edge_index, batch):

        # Perform 2 layers of message passing
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        # Global pooling
        h = global_max_pool(h, batch) # [num_examples, hidden_channels]

        # Classifer
        return self.classifier(h)
    
model = PointNet()

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        logits = model(data.pos, data.edge_index, data.batch)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test():
    model.eval()

    total_correct = 0
    for data in test_loader:
        logits = model(data.pos, data.edge_index, data.batch)
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(test_loader.dataset)

for epoch in range(1, 201):
    loss = train()
    test_acc = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}')

