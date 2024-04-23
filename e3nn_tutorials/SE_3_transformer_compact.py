import torch
import torch.nn.functional as F
import e3nn
from e3nn import o3
from e3nn.math import soft_one_hot_linspace, soft_unit_step
from torch_cluster import radius_graph
from torch_scatter import scatter

def transformer(f, pos, max_radius, number_of_basis, irreps_sh,
                h_q, tp_k, tp_v, fc_k, fc_v, dot):
    edge_src, edge_dst = radius_graph(pos, max_radius)
    edge_vec = pos[edge_src] - pos[edge_dst]
    edge_length = edge_vec.norm(dim=1)

    edge_length_embedded = soft_one_hot_linspace(edge_length, start=0.0, end=max_radius,
                                                 number=number_of_basis,
                                                 basis='smooth_finite', cutoff=True)
    edge_length_embedded = edge_length_embedded.mul(number_of_basis**0.5)
    edge_weight_cutoff = soft_unit_step(10*(1 - edge_length/max_radius))
    edge_sh = o3.spherical_harmonics(irreps_sh, edge_vec, normalize=True, 
                                     normalization='component')
    q = h_q(f)
    k = tp_k(f[edge_src], edge_sh, fc_k(edge_length_embedded))
    v = tp_v(f[edge_src], edge_sh, fc_v(edge_length_embedded))

    exp = edge_weight_cutoff[:,None] * dot(q[edge_dst], k).exp()
    z = scatter(exp, edge_dst, dim=0, dim_size=len(f))
    z[z==0] = 1
    alpha = exp / z[edge_dst]
    output = scatter(alpha.relu().sqrt()*v, edge_dst, dim=0, dim_size=len(f))
    return output

if __name__ == '__main__':
    number_of_basis = 10
    num_nodes = 20
    pos = torch.randn(num_nodes, 3)
    max_radius = 1.3

    irreps_input = o3.Irreps("10x0e + 5x1o + 2x2e")
    irreps_query = o3.Irreps("11x0e + 4x1o")
    irreps_key = o3.Irreps("12x0e + 3x1o")
    irreps_output = o3.Irreps("14x0e + 6x1o") # same irreps for the values

    f = irreps_input.randn(num_nodes, -1)

    # p = -1 for E(3), p = 1 for SE(3)
    irreps_sh = o3.Irreps.spherical_harmonics(3, p=-1) # parity for each l is (-1)**l

    h_q = o3.Linear(irreps_input, irreps_query) # think of the filter as a 1x0e

    tp_k = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_key, shared_weights=False)
    fc_k = e3nn.nn.FullyConnectedNet([number_of_basis, 16, tp_k.weight_numel], act=F.silu)

    tp_v = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_output, shared_weights=False)
    fc_v = e3nn.nn.FullyConnectedNet([number_of_basis, 16, tp_v.weight_numel], act=F.silu)

    dot = o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "1x0e")
    
    output = transformer(f, pos, max_radius, number_of_basis, irreps_sh,
                h_q, tp_k, tp_v, fc_k, fc_v, dot)
    
    # check equivariance
    rot = o3.rand_matrix()
    D_in = irreps_input.D_from_matrix(rot)
    D_out = irreps_output.D_from_matrix(rot)
    output_before = transformer(f @ D_in.T, pos @ rot.T, max_radius, number_of_basis, irreps_sh,
                                h_q, tp_k, tp_v, fc_k, fc_v, dot)
    output_after = output @ D_out.T
    torch.allclose(output_before, output_after, atol=1e-3, rtol=1e-3)
