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
z_full = nfm.integrate(x, tspan=[0.0, 1.0], nt=nt, intermediates=True)
print("tspan=[0, 1]")
print(np.around(z_full.data.numpy()[0, :, :].T, 3))

x = z_full[:, :d, -1]
z_full = nfm.integrate(x, tspan=[1.0, 0.0], nt=nt, intermediates=True)
print("tspan=[1, 0]")
print(np.round(z_full.data.numpy()[0, :, :].T, 3))