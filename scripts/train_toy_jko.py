"""Train OT-Flow on two-dimensional toy data using the JKO scheme.

OT-Flow is trained for a predetermined number of steps. After each step,
the target is updated to the pushforward of the learned density.
"""
import argparse
import copy
import logging
import math
import os
import pathlib
from pprint import pprint
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cnflows as cnf
import matplotlib.pyplot as plt
import numpy as np
import proplot as pplt
import torch

import plotting
from utils import get_torch_device
from utils import Monitor
from utils import ScriptManager

pplt.rc["grid"] = False
pplt.rc["cycle"] = "538"
pplt.rc["cmap.discrete"] = False
pplt.rc["cmap.sequential"] = "viridis"


# Parse command line arguments
# --------------------------------------------------------------------------------------
parser = argparse.ArgumentParser("ot-flow")

parser.add_argument("--save", type=int, default=1)
parser.add_argument("--outdir", type=str, default="./data_output/")
parser.add_argument("--data", type=str, default="swissroll")
parser.add_argument("--precision", type=str, default="single", choices=["single", "double"])
parser.add_argument("--gpu", type=int, default=0)

parser.add_argument("--m", type=int, default=16, help="network width")
parser.add_argument("--n_layers", type=int, default=2, help="network depth")
parser.add_argument("--alpha_L", type=float, default=1.0, help="loss function scaling (L)")
parser.add_argument("--alpha_C", type=float, default=10.0, help="loss function scaling (C)")
parser.add_argument("--alpha_R", type=float, default=1.0, help="loss function scaling (R)")

parser.add_argument("--data_size", type=int, default=int(1.00e+05), help="training data size")
parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
parser.add_argument("--nt", type=int, default=8, help="number of time steps")

parser.add_argument("--val_data_size", type=int, default=int(1.00e+05), help="validation - data size")
parser.add_argument("--val_batch_size", type=int, default=1024, help="validation - batch size")
parser.add_argument("--val_nt", type=int, default=8, help="validation - number of time steps")

parser.add_argument("--n_steps", type=int, default=5, help="number of subproblems")
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--max_lr", type=float, default=1.00e-02, help="initial learning rate")
parser.add_argument("--min_lr", type=int, default=1.00e-04, help="minimum learning rate")
parser.add_argument("--lr_drop_factor", type=int, default=0.1)
parser.add_argument("--lr_drop_patience", type=int, default=5)
parser.add_argument("--lr_drop_thresh", type=int, default=0.0001)
parser.add_argument("--early_stopping", type=int, default=0)
parser.add_argument("--avg_scaling", type=float, default=0.1, help="loss average scaling")
parser.add_argument("--weight_decay", type=float, default=0.0, help="ADAM weight decay")

parser.add_argument("--vis_freq", type=int, default=1, help="visualization frequency")
parser.add_argument("--vis_batch_size", type=int, default=int(10.00e+04), help="visualization batch size")

args = parser.parse_args()


# Settings
# --------------------------------------------------------------------------------------
device = get_torch_device(gpu=args.gpu)
precision = torch.float32
if args.precision == "double":
    precision = torch.float64
torch.set_default_dtype(precision)

man = ScriptManager(
    outdir=args.outdir,
    path=pathlib.Path(__file__), 
    use_prefix=False
)
man.save_script_copy()
pprint(man.get_info())

logger = man.get_logger(save=args.save, print=True, filename="log.txt")
for key, val in man.get_info().items():
    logger.info(f"{key}: {val}")
logger.info(args)
    

# Load training and data.
# --------------------------------------------------------------------------------------

d = 2  # number of dimensions
rng = None  # random number generator

def get_data(n):
    data = cnf.utils.gen_toy_data(name=args.data, size=n, rng=rng)
    data = torch.from_numpy(data)
    data = data.type(precision).to(device, non_blocking=True)
    return data


data = get_data(args.data_size)
dataset = torch.utils.data.TensorDataset(data)
data_loader = torch.utils.data.DataLoader(
    dataset=dataset, 
    batch_size=args.batch_size, 
    shuffle=False
)

val_data = get_data(args.val_data_size)
val_dataset = torch.utils.data.TensorDataset(val_data)
val_data_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, 
    batch_size=val_data.shape[0], 
    shuffle=False
)


# Model
# --------------------------------------------------------------------------------------
m = args.m
n_layers = args.n_layers
alpha = [args.alpha_L, args.alpha_C, args.alpha_R]

base_dist = cnf.distributions.DiagGaussian(d, trainable=False)
nfm = cnf.OTFlow(d=d, m=m, n_layers=n_layers, alpha=alpha, base_dist=base_dist)
nfm = nfm.to(precision).to(device)

logger.info("")
logger.info("Model")
logger.info("--------------------------------------------------------------------------")
logger.info(f"Number of trainable parameters = {nfm.get_n_params()}")
logger.info("")
for key, value in nfm.state_dict().items():
    logger.info(key)
    logger.info(value)
    logger.info("")


# Optimizer
# --------------------------------------------------------------------------------------
optimizer = torch.optim.Adam(
    nfm.potential.parameters(), 
    lr=args.max_lr, 
    weight_decay=args.weight_decay,
)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode="min", 
    factor=args.lr_drop_factor,
    patience=args.lr_drop_patience,
    threshold=args.lr_drop_thresh,
    threshold_mode="rel", 
    min_lr=args.min_lr, 
    verbose=False,
)

logger.info("")
logger.info("Optimizer")
logger.info("--------------------------------------------------------------------------")
logger.info(optimizer)


# Diagnostics
# --------------------------------------------------------------------------------------

def save_figure(filename, step=0, epoch=None, ext="png", **kwargs):
    filename = f"{filename}_{step:02.0f}"
    if epoch is not None:
        filename = f"{filename}_{epoch:05.0f}"
    filename = man.get_filename(filename)
    filename = f"{filename}.{ext}"
    kwargs.setdefault("dpi", 250)
    plt.savefig(filename, **kwargs)
    plt.close()
    return


# Training loop
# --------------------------------------------------------------------------------------

def create_data_loader(data_loader=None, nfm=None, nt=None):
    X_new = torch.zeros(1)
    with torch.no_grad():
        for (X,) in data_loader:
            X_temp, _, _, _ = nfm.forward(X, nt=nt)
            if X_new.shape[0] == 1:
                X_new = X_new
            else:
                X_new = torch.cat((X_new, X_temp), dim=0)
    dataset = torch.utils.data.TensorDataset(X_new)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=data_loader.batch_size,
        shuffle=False,
    )


state_dicts = []

for step in range(args.n_steps):
    print(f"step={step}")

    for param_group in optimizer.param_groups:
        param_group["lr"] = args.max_lr
    lr_scheduler.num_bad_epochs = 0

    best_avg_loss = float("inf")
    best_state_dict = nfm.state_dict()

    monitor = Monitor(
        outdir=man.outdir, 
        prefix=man.prefix, 
        filename=f"history_{step:02.0f}.dat", 
    )

    nfm.to(device)
    nfm.train()

    for epoch in range(args.n_epochs):
        avg_loss = 0.0
        for batch_index, (X,) in enumerate(data_loader):
            X = X.to(device)
            optimizer.zero_grad()
            loss, costs = nfm.forward_kld(X, nt=args.nt, return_costs=True)
            if not (torch.isinf(loss) or torch.isnan(loss)):
                loss.backward()
                optimizer.step()
            avg_loss = args.avg_scaling * loss + (1.0 - args.avg_scaling) * avg_loss
            monitor.action(
                epoch=epoch,
                lr=optimizer.param_groups[0]["lr"],
                loss=loss.detach().cpu().numpy(),
                loss_L=costs[0].detach().cpu().numpy(),
                loss_C=costs[1].detach().cpu().numpy(),
                loss_R=costs[2].detach().cpu().numpy(),
            )

        # Validation
        # [...]
        
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_state_dict = nfm.state_dict()

        if args.save and (epoch % args.vis_freq == 0 or epoch == args.n_epochs - 1):
            with torch.no_grad():
                nfm.eval()      
                current_state_dict = nfm.state_dict()
                nfm.load_state_dict(best_state_dict)

                history = monitor.get_data()
    
                X = torch.clone(val_data[:args.vis_batch_size, :])
                
                plotting.plot_samples(X, nfm, bins=50, xmax=4.5, nt=args.val_nt)
                save_figure("fig_samples", step=step, epoch=epoch)

                plotting.plot_loss_history(history, log=False)
                save_figure("fig__loss", step=step)

                plotting.plot_loss_history(history, log=True)
                save_figure("fig__loss_log", step=step)
                                
                fig, ax = pplt.subplots()
                ax.plot(history["iteration"], history["lr"], color="black")
                save_figure("fig__lr", step=step)
    
                nfm.load_state_dict(current_state_dict)
                nfm.train()

        lr_scheduler.step(avg_loss)
        
        stop_early = (
            args.early_stopping
            and optimizer.param_groups[0]["lr"] <= args.min_lr
            and lr_scheduler.num_bad_epochs > lr_scheduler.patience
        )
        if stop_early:
            print("Stopping early: lr <= min_lr and num_bad_epochs > patience")
            break
                
    # Update target.
    print("Updating target")
    nfm.load_state_dict(best_state_dict)
    data_loader = create_data_loader(data_loader, nfm, args.nt)
    val_data_loader = create_data_loader(val_data_loader, nfm, args.nt)
    val_data = next(iter(val_data_loader))[0]

    # Save the model.
    print("Saving model")
    state_dict = copy.deepcopy(nfm.state_dict())
    state_dicts.append(state_dict)
    state = {
        "state_dict": state_dict,
        "d": d,
        "m": m,
        "alpha": alpha,
        "n_layers": n_layers,
        "base_dist": nfm.base_dist,
    }
    filename = man.get_filename(f"state_dict_{step}.pth")
    torch.save(state, filename)

    # Evaluate chained models.
    print("Plotting")
    with torch.no_grad():
        current_state_dict = nfm.state_dict()
        nfm.eval()

        X_true = get_data(args.vis_batch_size)

        plotting.plot_samples_chained(X_true, nfm=nfm, state_dicts=state_dicts, nt=args.val_nt)
        save_figure("fig__loss", step=step)
        
        nfm.load_state_dict(current_state_dict)
        nfm.train()
    
    