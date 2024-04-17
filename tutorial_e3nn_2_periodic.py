"""
https://docs.e3nn.org/en/latest/guide/periodic_boundary_conditions.html
"""

import torch
import e3nn
import ase
import ase.neighborlist
import torch_geometric
import torch_geometric.data

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

# A lattice is a 3x3 matrix
# The first index is the lattice vector (a,b,c)
# The second index is a Cartesian index over (x,y,z)

# Polonium with Simple Cubic Lattice
po_lattice = torch.eye(3) * 3.340 # cubic lattice with edges of length 3.34 AA
po_coords = torch.tensor([[0., 0., 0.]])
po_types = ['Po']

# Silicon with Diamond Structure
si_lattice = torch.tensor([[0., 2.734364, 2.734364], 
                           [2.734364, 0., 2.734364], 
                           [2.734364, 2.734364, 0.]])
si_coords = torch.tensor([[1.367182, 1.367182, 1.367182],
                          [0., 0., 0.]])
si_types = ['Si', 'Si']

po = ase.Atoms(symbols=po_types, positions=po_coords, cell=po_lattice, pbc=True)
si = ase.Atoms(symbols=si_types, positions=si_coords, cell=si_lattice, pbc=True)

radial_cutoff = 3.5 # only include edges for neighboring atoms within a radius of 3.5 AA
type_encoding = {'Po': 0, 'Si': 1}
type_onehot = torch.eye(len(type_encoding))

dataset = []
dummy_energies = torch.randn(2, 1, 1)

for crystal, energy in zip([po, si], dummy_energies):
    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images / copies of the unit cell
    edge_src, edge_dst, edge_shift = ase.neighborlist.neighbor_list("ijS", a=crystal, cutoff=radial_cutoff, self_interaction=True)
    data = torch_geometric.data.Data(
        pos=torch.tensor(crystal.get_positions()),
        lattice=torch.tensor(crystal.cell.array).unsqueeze(0), # add a batch dimension
        x=type_onehot[[type_encoding[atom] for atom in crystal.symbols]],
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        energy=energy # dummy energy
    )
    dataset.append(data)

print(dataset)

# graph batches

batch_size = 2
dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=batch_size)

for data in dataloader:
    print("Data")
    print(data)
    print(data.batch)
    print(data.pos)
    print(data.x)

edge_src, edge_dst = data['edge_index'][0], data['edge_index'][1]

edge_vec = (data['pos'][edge_dst] - data['pos'][edge_src] + torch.einsum('ni,nij->nj', data['edge_shift'], data['lattice'][data.batch[edge_src]]))

from e3nn.nn.models.v2103.gate_points_networks import SimpleNetwork
import torch_scatter

class SimplePeriodicNetwork(SimpleNetwork):
    def __init__(self, **kwargs):
        """
        SimpleNetwork uses the keyword 'pool_nodes' to determine whether we sum over all atom contributions per example.
        Here, we use mean operations, so we override this behavior
        """
        self.pool = False
        if kwargs['pool_nodes'] == True:
            kwargs['pool_nodes'] = False
            kwargs['num_nodes'] = 1
            self.pool = True
        super().__init__(**kwargs)

    # overwrite preprocess method of SimpleNetwork
    def preprocess(self, data):
        batch = data['batch']
        edge_src, edge_dst = data['edge_index']
        edge_batch = batch[edge_src]
        edge_vec = (data['pos'][edge_dst] - data['pos'][edge_src] 
                    + torch.einsum('ni,nij->nj', data['edge_shift'], data['lattice'][edge_batch]))
        return batch, data['x'], edge_src, edge_dst, edge_vec
    
    def forward(self, data):
        output = super().forward(data)
        if self.pool == True:
            return torch_scatter.scatter_mean(output, data.batch, dim=0)
        else:
            return output

net = SimplePeriodicNetwork(irreps_in="2x0e", irreps_out="1x0e", max_radius=radial_cutoff, num_neighbors=10.0, pool_nodes=True)
for data in dataloader:
    print(net(data))
