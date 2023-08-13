import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cnflows as cnf
import matplotlib.pyplot as plt
import numpy as np
import proplot as pplt
import torch

sys.path.append("/Users/46h/repo/ot-flow/")
from src.OTFlowProblem import OTFlowProblem
from src.OTFlowProblem import integrate
from src.Phi import Phi


# Settings
save = True
prec = torch.float32
device = torch.device("cpu")
torch.set_default_dtype(prec)
cvt = lambda x: x.type(prec).to(device, non_blocking=True)

pplt.rc["grid"] = False
pplt.rc["cmap.discrete"] = False
pplt.rc["cmap.sequential"] = "viridis"


# Set 2D target distribution.
d = 2  # dimensionality of data
target_dist = cnf.distributions.TwoMoons()
base_dist = cnf.distributions.DiagGaussian(2)


# Neural network for the potential function Phi
d = 2
alpha = [1.0, 100.0, 15.0]
nTh = 2
m = 32
net = Phi(nTh=nTh, m=m, d=d, alph=alpha)
net = net.to(prec).to(device)


# Create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=0.0)


def compute_loss(net, x, nt):
    Jc, cs = OTFlowProblem(x, net, [0.0, 1.0], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs

n_iterations = 2000
n_samples = 512
nt = 8
best_loss = torch.tensor(1.00e+12)

net.train()
for iteration in range(n_iterations):
    optimizer.zero_grad()
    X = cvt(target_dist.sample(n_samples))
    loss, costs = compute_loss(net, X, nt=nt)
    loss.backward()
    optimizer.step()

    if loss < best_loss:
        best_loss = loss
        best_state_dict = net.state_dict()

    print(f"{iteration}, loss={loss:.2e}, L={costs[0]:.2e}, C={costs[1]:.2e}, R={costs[2]:.2e}")

    if save and (iteration % 100 == 0):
        with torch.no_grad():
            net.eval()      
            state_dict = net.state_dict()
            net.load_state_dict(best_state_dict)

            n_bins = 150
            xgrid = torch.linspace(-3.5, 3.5, n_bins)
            COORDS = torch.meshgrid(xgrid, xgrid, indexing="ij")
            X_grid = torch.vstack([C.ravel() for C in COORDS]).T
            X_grid = cvt(X_grid)
            
            z = integrate(X_grid, net, [0.0, 1.0], nt, stepper="rk4", alph=net.alph)
            xn = z[:, :d]
            log_det = z[:, d]
            log_prob = log_det + base_dist.log_prob(xn)
            prob = torch.exp(log_prob)
            prob = prob.data.numpy()
            prob = prob.reshape((n_bins, n_bins))

            log_prob_target = target_dist.log_prob(X_grid)
            prob_target = torch.exp(log_prob_target)
            prob_target = prob_target.data.numpy()
            prob_target = prob_target.reshape((n_bins, n_bins))
            
            fig, axs = pplt.subplots(ncols=2)
            axs[0].pcolormesh(xgrid.data.numpy(), xgrid.data.numpy(), prob_target.T)
            axs[1].pcolormesh(xgrid.data.numpy(), xgrid.data.numpy(), prob.T)
            plt.savefig(f"figures/fig_ot_{iteration:04.0f}.png", dpi=350)
            plt.close()

            net.load_state_dict(state_dict)
            
            net.train()
