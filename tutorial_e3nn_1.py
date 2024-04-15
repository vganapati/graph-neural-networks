import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace
import matplotlib.pyplot as plt

irreps_input = o3.Irreps("20x0e + 10x1e") 
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

fc = nn.FullyConnectedNet([num_basis, 16, tp.weight_numel], torch.relu)
weight = fc(edge_length_embedding)

summand = tp(f_in[edge_src], sh, weight)
f_out = scatter(summand, edge_dst, dim=0, dim_size=num_nodes)
f_out = f_out.div(num_neighbors**0.5)


def conv(f_in, pos):
    edge_src, edge_dst = radius_graph(pos, max_radius, max_num_neighbors=len(pos) - 1)
    edge_vec = pos[edge_dst] - pos[edge_src]
    sh = o3.spherical_harmonics(irreps_sh, edge_vec, normalize=True, normalization='component')
    emb = soft_one_hot_linspace(edge_vec.norm(dim=1), 0.0, max_radius, num_basis, basis='smooth_finite', cutoff=True).mul(num_basis**0.5)
    return scatter(tp(f_in[edge_src], sh, fc(emb)), edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5)

# check equivariance
rot = o3.rand_matrix()
D_in = irreps_input.D_from_matrix(rot)
D_out = irreps_output.D_from_matrix(rot)

# rotate before
f_before = conv(f_in @ D_in.T, pos @ rot.T)

# rotate after
f_after = conv(f_in, pos) @ D_out.T

assert torch.allclose(f_before, f_after, rtol=1e-4, atol=1e-4)

# timing of the different elements

import time
wall = time.perf_counter()

edge_src, edge_dst = radius_graph(pos, max_radius, max_num_neighbors=len(pos)-1)
edge_vec = pos[edge_dst] - pos[edge_src]
print(time.perf_counter() - wall); wall = time.perf_counter()

sh = o3.spherical_harmonics(irreps_sh, edge_vec, normalize=True, normalization='component')
print(time.perf_counter()-wall); wall = time.perf_counter()

emb = soft_one_hot_linspace(edge_vec.norm(dim=1), 0.0, max_radius, num_basis, basis='smooth_finite', cutoff=True).mul(num_basis**0.5)
print(time.perf_counter()-wall); wall = time.perf_counter()

weight = fc(emb)
print(time.perf_counter()-wall); wall = time.perf_counter()

summand = tp(f_in[edge_src], sh, weight)
print(time.perf_counter()-wall); wall = time.perf_counter()

scatter(summand, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5)
print(time.perf_counter()-wall); wall = time.perf_counter()