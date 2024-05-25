"""
Shallow Node Embeddings tutorial, node2vec
Reference: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/shallow_node_embeddings.html

Nearby nodes should receive similar embeddings while distant nodes should receive distinct embeddings

Give each node an embedding z of some length. Take a random walk of length k starting from a node n. We have the set of nodes in the walk, 
and the set of nodes not in the walk. We want the embeddings for the nodes in the walk to be similar to to n, and we don't
want the nodes not in the walk to be similar.

Train the embedding for node n with the following loss:
Loss = nodes in walk i (-log(sigmoid(similarity of n and node i))) + nodes not in walk i (-log(1 - sigmoid(similarity of n and node i)))

Use the trained embeddings z for downstream tasks. Edge-level representations can be obtained by combined the embeddings of the 
start and end nodes.

Use case: the embeddings are tied to a particular graph. Useful for cases where the initial set of node features is not very rich.
"""

import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = Planetoid('./data/Planetoid', name='Cora')[0]

model = Node2Vec(data.edge_index,
                 embedding_dim=128,
                 walks_per_node=10,
                 walk_length=20,
                 context_size=10, # how many nodes in the walk are used for gradient calculations
                 p=1.0, # likelihood of immediately revisiting a node
                 q=1.0, # breadth first vs. depth first
                 num_negative_samples=1, # how many negative walks for each positive walk
                 ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loader = model.loader(batch_size=128, shuffle=True, num_workers=0)

"""
shape of pos_rw is [batch_size*walks_per_node*(1 + walk_length - context_size), context_size]
shape of neg_rw is the same except 0 dimension is multiplied by num_negative_samples
"""

for pos_rw, neg_rw in loader:
    print(pos_rw.shape)

def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / data.x.shape[0]

z = model()
print(sum(z[0]*z[633]))
print(sum(z[2707]*z[598]))

for epoch in range(10):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

z = model()
print(sum(z[0]*z[633]))
print(sum(z[2707]*z[598]))
breakpoint()



