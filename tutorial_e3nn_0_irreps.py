"""
https://docs.e3nn.org/en/latest/guide/irreps.html
"""
from e3nn import o3
import torch
import numpy as np
import matplotlib.pyplot as plt

irreps = o3.Irreps("1e")
print(irreps)

t = torch.tensor
print(irreps.D_from_angles(alpha=t(np.pi), beta=t(0.0), gamma=t(0.0), k=t(0)))
irreps = o3.Irreps("10x0e + 5x1e + 5x2e")

rot = o3.rand_matrix()
D = irreps.D_from_matrix(rot)
plt.figure()
plt.imshow(D, cmap='bwr', vmin=-1, vmax=1)
plt.savefig('Wigner_D_matrix.png')
