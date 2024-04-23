"""
Reference: https://docs.e3nn.org/en/stable/guide/transformer.html
"""

import torch
import e3nn
from e3nn import o3
from e3nn.math import soft_one_hot_linspace, soft_unit_step
from torch_cluster import radius_graph
from torch_scatter import scatter
import matplotlib.pyplot as plt
# define arbitrary irreps

irreps_input = o3.Irreps("10x0e + 5x1o + 2x2e")
irreps_query = o3.Irreps("11x0e + 4x1o")
irreps_key = o3.Irreps("12x0e + 3x1o")
irreps_output = o3.Irreps("14x0e + 6x1o") # same irreps for values

# create random graph

num_nodes = 20
pos = torch.randn(num_nodes, 3)
f = irreps_input.randn(num_nodes, -1)

max_radius = 1.3
edge_src, edge_dst = radius_graph(pos, max_radius)
edge_vec = pos[edge_src] - pos[edge_dst]
edge_length = edge_vec.norm(dim=1)

# queries q_i are a linear combination of the input features f_i
h_q = o3.Linear(irreps_input, irreps_query)

""" 
# same number of weights in the following:
irreps_sh = o3.Irreps.spherical_harmonics(0)
h_q = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_query, shared_weights=False)
"""

# generate weights that depend on distance

number_of_basis = 10
edge_length_embedded = soft_one_hot_linspace(edge_length, start=0.0, end=max_radius,
                                             number=number_of_basis, basis='smooth_finite',
                                             cutoff=True)
edge_weight_cutoff = soft_unit_step(10*(1 - edge_length/max_radius))

irreps_sh = o3.Irreps.spherical_harmonics(3)
edge_sh = o3.spherical_harmonics(irreps_sh, edge_vec, True, normalization='component')

tp_k = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_key, shared_weights=False)
fc_k = e3nn.nn.FullyConnectedNet([number_of_basis, 16, tp_k.weight_numel],act=torch.nn.functional.silu)

tp_v = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_output, shared_weights=False)
fc_v = e3nn.nn.FullyConnectedNet([number_of_basis, 16, tp_v.weight_numel], act=torch.nn.functional.silu)

dot = o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "0e")

# compute the queries (per node), keys (per edge), and values (per edge)
q = h_q(f)
k = tp_k(f[edge_src], edge_sh, fc_k(edge_length_embedded))
v = tp_v(f[edge_src], edge_sh, fc_v(edge_length_embedded))

# compute the softmax (per edge)
exp = edge_weight_cutoff[:, None] * dot(q[edge_dst], k).exp()
z = scatter(exp, edge_dst, dim=0, dim_size=len(f))
z[z==0] = 1 # to avoid 0/0 when all the neighbors are exactly at the cutoff
alpha = exp/z[edge_dst]

# compute the outputs (per node)
f_out = scatter(alpha.relu().sqrt()*v, edge_dst, dim=0, dim_size=len(f))

def transformer(f, pos, use_edge_weight_cutoff=True):
    edge_src, edge_dst = radius_graph(pos, max_radius)
    edge_vec = pos[edge_src] - pos[edge_dst]
    edge_length = edge_vec.norm(dim=1)

    edge_length_embedded = soft_one_hot_linspace(edge_length, start=0.0, end=max_radius, 
                                                 number=number_of_basis, 
                                                 basis='smooth_finite', cutoff=True)
    edge_length_embedded = edge_length_embedded.mul(number_of_basis**0.5)
    edge_weight_cutoff = soft_unit_step(10*(1 - edge_length/max_radius))
    edge_sh = o3.spherical_harmonics(irreps_sh, edge_vec, True, normalization='component')

    q = h_q(f)
    k = tp_k(f[edge_src], edge_sh, fc_k(edge_length_embedded))
    v = tp_v(f[edge_src], edge_sh, fc_v(edge_length_embedded))

    if use_edge_weight_cutoff:
        exp = edge_weight_cutoff[:,None] * dot(q[edge_dst], k).exp()
    else:
        exp = dot(q[edge_dst], k).exp()

    z = scatter(exp, edge_dst, dim=0, dim_size=len(f))
    z[z==0] = 1
    alpha = exp / z[edge_dst]
    output = scatter(alpha.relu().sqrt()*v, edge_dst, dim=0, dim_size=len(f))
    return output

# smoothness check
# 2 nodes "see" each other, a third node coming from far away moves slowly towards them

f = irreps_input.randn(3,-1)
xs = torch.linspace(-1.3, -1.0, 200)
outputs = []

for x in xs:
    pos = torch.tensor([
        [0.0, 0.5, 0.0], # this node always sees...
        [0.0, -0.5, 0.0], # ...this node
        [x.item(), 0.0, 0.0], # this node moves slowly
    ])

    with torch.no_grad():
        outputs.append(transformer(f, pos, use_edge_weight_cutoff=True))

outputs = torch.stack(outputs)
plt.figure()
plt.plot(xs, outputs[:, 0, [0, 1, 14, 15, 16]], 'k')
plt.plot(xs, outputs[:, 1, [0, 1, 14, 15, 16]], 'g')
plt.plot(xs, outputs[:, 2, [0, 1, 14, 15, 16]], 'r')

plt.xlabel('3rd node position')
plt.ylabel('output features')
plt.savefig('distance_convergence.png')

# check equivariance

f = irreps_input.randn(10,-1)
pos = torch.randn(10, 3)

rot = o3.rand_matrix()
D_in = irreps_input.D_from_matrix(rot)
D_out = irreps_output.D_from_matrix(rot)

f_before = transformer(f @ D_in.T, pos @ rot.T)
f_after = transformer(f, pos) @ D_out.T

torch.allclose(f_before, f_after, atol=1e-3, rtol=1e-3)

# check the backwards pass

for x in [0.0, 1e-6, max_radius/2, max_radius - 1e-6, max_radius, max_radius + 1e-6, 2*max_radius]:
    f = irreps_input.randn(3, -1, requires_grad=True)
    pos = torch.tensor([[0.0, 0.0, 0.0], [x, 0.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=True)
    transformer(f, pos, use_edge_weight_cutoff=True).sum().backward()
    assert f.grad is None or torch.isfinite(f.grad).all()
    assert torch.isfinite(pos.grad).all()
