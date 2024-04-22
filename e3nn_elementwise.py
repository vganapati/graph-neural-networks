import matplotlib.pyplot as plt
from e3nn import o3

tp = o3.ElementwiseTensorProduct(
    irreps_in1='32x0o',
    irreps_in2='16x1e+16x1o'
)

print(tp)
plt.figure()
tp.visualize()
plt.savefig("elementwise.png")