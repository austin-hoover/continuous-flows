import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cnflows as cnf
import matplotlib.pyplot as plt
import numpy as np
import proplot as pplt
import torch


# Settings
save = False
precision = torch.float32
device = torch.device("cpu")
torch.set_default_dtype(precision)
cvt = lambda x: x.type(precision).to(device, non_blocking=True)

pplt.rc["grid"] = False
pplt.rc["cmap.discrete"] = False
pplt.rc["cmap.sequential"] = "viridis"


# Set 2D target distribution.
d = 2  # dimensionality of data
target_dist = cnf.distributions.TwoMoons()
base_dist = cnf.distributions.DiagGaussian(2)


# Create normalizing flow model.
d = 2
alpha = [1.0, 100.0, 5.0]
n_layers = 2
m = 16
nfm = cnf.OTFlow(d=d, n_layers=n_layers, m=m, alpha=alpha, base_dist=None)
nfm = nfm.to(precision).to(device)


# Create optimizer
optimizer = torch.optim.Adam(nfm.parameters(), lr=0.01, weight_decay=0.0)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode="min",
    factor=0.33, 
    patience=100, 
    threshold=0.0001, 
    threshold_mode="rel",
    cooldown=0, 
    min_lr=1.00e-04, 
    verbose=False
)

# Train the model.
n_iterations = 2000
n_samples = 512
nt = 8
best_loss = torch.tensor(1.00e+12)

nfm.train()
for iteration in range(n_iterations):
    optimizer.zero_grad()
    X = cvt(target_dist.sample(n_samples))
    loss, costs = nfm.forward_kld(X, nt=8, return_costs=True)
    loss.backward()
    optimizer.step()

    lr = optimizer.param_groups[0]["lr"]
    print(f"{iteration}, lr={lr:.2e}, loss={loss:.2e}, L={costs[0]:.2e}, C={costs[1]:.2e}, R={costs[2]:.2e}, best_loss={best_loss:.2e}")

    if loss < best_loss:
        best_loss = loss
        best_state_dict = nfm.state_dict()

    if save and (iteration % 100 == 0):
        with torch.no_grad():
            nfm.eval()      
            state_dict = nfm.state_dict()
            nfm.load_state_dict(best_state_dict)

            n_bins = 150
            xgrid = torch.linspace(-3.5, 3.5, n_bins)
            COORDS = torch.meshgrid(xgrid, xgrid, indexing="ij")
            X_grid = torch.vstack([C.ravel() for C in COORDS]).T
            X_grid = cvt(X_grid)

            log_prob = nfm.log_prob(X_grid, nt=nt)
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
            plt.savefig(f"figures/fig_{iteration:04.0f}.png", dpi=350)
            plt.close()

            nfm.load_state_dict(state_dict)
            nfm.train()
            
    lr_scheduler.step(loss)