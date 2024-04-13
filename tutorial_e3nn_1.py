import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace
import matplotlib.pyplot as plt

irreps_input = o3.Irreps("10x0e + 10x1e")
irreps_output = o3.Irreps("10x0e + 10x1e")


# create node positions
num_nodes = 100
pos = torch.randn(num_nodes, 3) # random node positions

# create edges
max_radius = 1.8
edge_src, edge_dst = radius_graph(pos, max_radius, max_num_neighbors=num_nodes-1)
print(edge_src.shape)
edge_vec = pos[edge_dst] - pos[edge_src]

# compute z
num_neighbors = len(edge_src) / num_nodes
print(num_neighbors)

f_in = irreps_input.randn(num_nodes, -1)

irreps_sh = o3.Irreps.spherical_harmonics(lmax=2, p=1)
sh = o3.spherical_harmonics(irreps_sh, edge_vec, normalize=True, normalization='component')

tp = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_output, shared_weights=False)
plt.figure()
tp.visualize()
plt.savefig("connections.png")

num_basis = 10
x = torch.linspace(0.0, 2.0, 1000)
y = soft_one_hot_linspace(x, start=0.0, end=max_radius, number=num_basis, basis='smooth_finite', cutoff=True)
plt.figure()
plt.plot(x,y)
plt.savefig('base_functions.png')

edge_length_embedding = soft_one_hot_linspace(edge_vec.norm(dim=1), start=0.0, end=max_radius, number=num_basis,
                                              basis='smooth_finite', cutoff=True)
edge_length_embedding = edge_length_embedding.mul(num_basis**0.5)


breakpoint()