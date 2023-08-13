import sys

import numpy as np
import torch

sys.path.append("..")
import cnflows as cnf


d = 2  # dimensionality of data
nt = 6  # number of integration time steps
m = 16  # number of hidden dimensions (width)
n_layers = 2  # number of layers (depth)
base_dist = cnf.distributions.DiagGaussian(d)
nfm = cnf.OTFlow(
    d=d,
    m=m,
    n_layers=n_layers,
    alpha=[1.0, 1.0, 1.0],
    base_dist=base_dist,
)

x = torch.tensor([[1.0, -1.0],])
print("x =", x[0].data.numpy())

output = nfm.integrate(x, tspan=[0.0, 1.0], nt=nt, intermediates=True)
z = output[:, :d, :]
l = output[:, d, :]
v = output[:, d + 1, :]
r = output[:, d + 2, :]

print()
print()
print("z =", z[0].data.numpy())
print()
print("l =", l[0].data.numpy())
print()
print("v =", v[0].data.numpy())
print()
print("r =", r[0].data.numpy())

z = z[:, :d, -1]
output = nfm.integrate(z, tspan=[1.0, 0.0], nt=nt, intermediates=True)
x = output[:, :d, :]
l = output[:, d, :]
v = output[:, d + 1, :]
r = output[:, d + 2, :]

print()
print()
print("x =", x[0].data.numpy())
print()
print("l =", l[0].data.numpy())
print()
print("v =", v[0].data.numpy())
print()
print("r =", r[0].data.numpy())
