import e3nn.o3
from e3nn.util.test import equivariance_error
from e3nn.util.test import assert_equivariant

# XXX Shouldn't this test fail for parity == 1 and succeed for parity == -1?
lmax = 2
parity = 1

irreps_input = e3nn.o3.Irreps("2x0e + 3x1o")
irreps_output = e3nn.o3.Irreps("2x1o")

irreps_sh = e3nn.o3.Irreps.spherical_harmonics(lmax, p=parity)
tp = e3nn.o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_output)

print(equivariance_error(tp, 
                         args_in=[tp.irreps_in1.randn(1, -1), tp.irreps_in2.randn(1,-1)], 
                         irreps_in=[tp.irreps_in1, tp.irreps_in2], 
                         irreps_out=[tp.irreps_out],
                         do_parity=True,
                         do_translation=True))

print(assert_equivariant(tp))
