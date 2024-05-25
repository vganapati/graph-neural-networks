"""
Irreducible representations
"""

import torch
from e3nn.o3 import Irreps, rand_matrix, spherical_harmonics, FullyConnectedTensorProduct, Linear
import matplotlib.pyplot as plt

torch.manual_seed(42)

irreps = Irreps("0o")
parity = irreps[0][1][1]

# transformation matrix for identity
D_identity = irreps.D_from_angles(alpha=torch.tensor(0.0), beta=torch.tensor(0.0), gamma=torch.tensor(0.0), k=torch.tensor(0))

# transformation matrix for inversion
D_inversion = irreps.D_from_angles(alpha=torch.tensor(0.0), beta=torch.tensor(0.0), gamma=torch.tensor(0.0), k=torch.tensor(1))

# transformation matrix for rotation
D_rotation = irreps.D_from_angles(alpha=torch.tensor(2.0), beta=torch.tensor(0.0), gamma=torch.tensor(0.0), k=torch.tensor(0))

irreps = Irreps("7x0e + 3x0o + 1x1e + 5x1o + 5x2o")
rot = rand_matrix() # rotation
rot = -rot # inversion
D = irreps.D_from_matrix(rot)

plt.figure()
plt.imshow(D, cmap='bwr', vmin=-1, vmax=1)
plt.savefig('D_mat_visualize.png')


"""
Convolution

source node output = 1/(sqrt(# of neighbors)) sum over neighbors [feature (tensor product) spherical harmonic(unit vector)*MLP(distance) ]
"""

from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import nn
from e3nn.math import soft_one_hot_linspace, soft_unit_step

irreps_input = Irreps("10x0e + 10x1e + 10x1o + 2x2o")
irreps_output = Irreps("20x0e + 10x1e + 3x1o + 1x1o + 2x2o")

parity = -1
irreps_sh = Irreps.spherical_harmonics(lmax=2, p=parity)

# create node positions
num_nodes = 100
pos = torch.randn(num_nodes, 3) # random node positions

# create edges
max_radius = 1.8
edge_src, edge_dest = radius_graph(pos, max_radius, max_num_neighbors=num_nodes-1)

# edge_src is the starting node of each edge
# edge_dest is the ending node (destination node) of each edge

edge_vec = pos[edge_dest] - pos[edge_src] # vector distance

# compute z, the number of neighbors (average node degree)
num_neighbors = len(edge_src) / num_nodes

# create some random input features
f_in = irreps_input.randn(num_nodes, -1)

# normalize=True ensures that x is divided by |x| before computing sh
# normalization=component is normalization of the output by dividing by sqrt(2*l + 1)
# normalization is valid b/c we are multiplying all the spherical harmonics of order l with the same thing
sh = spherical_harmonics(irreps_sh, edge_vec, normalize=True, normalization='component')

# tensor product

tp = FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_output, shared_weights=False)
print(f"{tp} needs {tp.weight_numel} weights")

plt.figure()
tp.visualize()
plt.savefig("tensor_product_visualization.png")

# create distance embedding

num_basis = 10
x = torch.linspace(0.0, 2.0, 1000)
y = soft_one_hot_linspace(x, start=0.0, end=max_radius, number=num_basis, basis='smooth_finite', cutoff=True)

plt.figure()
plt.plot(x,y)
plt.savefig('distance_embedding.png')

edge_length_embedding = soft_one_hot_linspace(edge_vec.norm(dim=1), start=0.0, end=max_radius, number=num_basis, basis="smooth_finite", cutoff=True)

edge_length_embedding = edge_length_embedding*(num_basis**0.5) # get variance close to 1

# create MLP with input distance embedding and output the filter weights

fc = nn.FullyConnectedNet([num_basis, 16, tp.weight_numel], torch.relu)
weight = fc(edge_length_embedding)

summand = tp(f_in[edge_src], sh, weight)

# sum over neighbors

f_out = scatter(summand, edge_dest, dim=0, dim_size=num_nodes)
f_out = f_out.div(num_neighbors**0.5)

# make a function
def conv(f_in, pos):
    edge_src, edge_dest = radius_graph(pos, max_radius, max_num_neighbors=len(pos)-1)
    edge_vec = pos[edge_dest] - pos[edge_src]
    sh = spherical_harmonics(irreps_sh, edge_vec, normalize=True, normalization='component')
    emb = soft_one_hot_linspace(edge_vec.norm(dim=1), 0.0, max_radius, num_basis, basis='smooth_finite', cutoff=True).mul(num_basis**0.5)
    return scatter(tp(f_in[edge_src], sh, fc(emb)), edge_dest, dim=0, dim_size=num_nodes).div(num_neighbors**0.5)

def test_equivariance(rot):
    D_in = irreps_input.D_from_matrix(rot)
    D_out = irreps_output.D_from_matrix(rot)

    # rotate before
    f_before = conv(f_in @ D_in.T, pos @ rot.T)

    # rotate after
    f_after = conv(f_in, pos) @ D_out.T

    print(torch.allclose(f_before, f_after, rtol=1e-4, atol=1e-4))

# check equivariance
rot = rand_matrix()
test_equivariance(rot)
test_equivariance(-rot)

"""
SE(3) Transformer
"""

irreps_input = Irreps("10x0e + 5x1o + 2x2e")
irreps_query = Irreps("11x0e + 4x1o")
irreps_key = Irreps("12x0e + 3x1o")
irreps_output = Irreps("14x0e + 6x1o") # also irreps of the values
irreps_sh = Irreps.spherical_harmonics(3)

# create graph

num_nodes = 20
pos = torch.randn(num_nodes, 3)
f = irreps_input.randn(num_nodes, -1)

max_radius = 1.3
edge_src, edge_dest = radius_graph(pos, max_radius)
edge_vec = pos[edge_src] - pos[edge_dest]
edge_length = edge_vec.norm(dim=1)


number_of_basis = 10
edge_length_embedded = soft_one_hot_linspace(edge_length, start=0.0, end=max_radius, 
                                             number=number_of_basis, basis='smooth_finite', cutoff=True)
edge_length_embedded = edge_length_embedded*(number_of_basis**0.5)
edge_weight_cutoff = soft_unit_step(10*(1 - edge_length/max_radius))

edge_sh = spherical_harmonics(irreps_sh, edge_vec, True, normalization="component")

h_q = Linear(irreps_input, irreps_query)
tp_q = FullyConnectedTensorProduct(irreps_input,"1x0e",irreps_query, shared_weights=False)

tp_k = FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_key, shared_weights=False)
fc_k = nn.FullyConnectedNet([number_of_basis, 16, tp_k.weight_numel], act=torch.nn.functional.silu)

tp_v = FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_output, shared_weights=False)
fc_v = nn.FullyConnectedNet([number_of_basis, 16, tp_v.weight_numel], act=torch.nn.functional.silu)

dot = FullyConnectedTensorProduct(irreps_query, irreps_key, "0e")

# compute the queries (per node), keys (per edge), and values (per edge)
q = h_q(f) # 20 x 23
q_0 = tp_q(f, torch.ones([num_nodes,1]), h_q.weight*torch.ones([num_nodes,tp_q.weight_numel])) # equivalent way to get q
k = tp_k(f[edge_src], edge_sh, fc_k(edge_length_embedded)) # inputs, filters, weights
v = tp_v(f[edge_src], edge_sh, fc_v(edge_length_embedded))

# compute the softmax (per edge)
exp = edge_weight_cutoff[:, None] * dot(q[edge_dest], k).exp() # compute the numerator
z = scatter(exp, edge_dest, dim=0, dim_size=len(f)) # compute the denominator
z[z==0] = 1 # to avoid 0/0 when all the neighbors are exactly at the cutoff
alpha = exp / z[edge_dest]

# compute the outputs (per node)
f_out = scatter(alpha.relu().sqrt()*v, edge_dest, dim=0, dim_size=len(f))