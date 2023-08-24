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

from adabelief_pytorch import AdaBelief
import matplotlib.pyplot as plt
import numpy as np
import proplot as pplt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cnflows as cnf

import plotting
from utils import get_torch_device_mps
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
parser.add_argument("--data", type=str, default="2spirals")
parser.add_argument("--precision", type=str, default="single", choices=["single", "double"])
parser.add_argument("--gpu", type=int, default=0)

parser.add_argument("--m", type=int, default=16)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--alpha_L", type=float, default=1.0)
parser.add_argument("--alpha_C", type=float, default=10.0)
parser.add_argument("--alpha_R", type=float, default=1.0)

parser.add_argument("--data_size", type=int, default=int(1.00e05))
parser.add_argument("--batch_size", type=int, default=2000)
parser.add_argument("--nt", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--val_nt", type=int, default=8)

parser.add_argument("--n_steps", type=int, default=5)
parser.add_argument("--n_epochs", type=int, default=40)

parser.add_argument("--optim", type=str, default="adam", choices=["adam", "adabelief"])
parser.add_argument("--lr", type=float, default=1.00e-02)
parser.add_argument("--min_lr", type=float, default=1.00e-03)
parser.add_argument("--lr_drop", type=float, default=0.5)
parser.add_argument("--lr_patience", type=int, default=5)

parser.add_argument("--vis_freq", type=int, default=2)
parser.add_argument("--vis_n_points", type=int, default=int(10.00e+04))
parser.add_argument("--vis_n_bins", type=int, default=100)
parser.add_argument("--vis_cmap", type=str, default="viridis")

args = parser.parse_args()


# Settings
# --------------------------------------------------------------------------------------
device = get_torch_device_mps(gpu=args.gpu)

precision = torch.float32
if args.precision == "double":
    precision = torch.float64
torch.set_default_dtype(precision)

cvt = lambda x: x.type(precision).to(device, non_blocking=True)

man = ScriptManager(outdir=args.outdir, path=pathlib.Path(__file__))
man.save_script_copy()
pprint(man.get_info())

logger = man.get_logger(save=args.save, print=True, filename="log.txt")
for key, val in man.get_info().items():
    logger.info(f"{key}: {val}")
logger.info(args)


# Load training data.
# --------------------------------------------------------------------------------------


def get_data(size):
    data = cnf.utils.gen_toy_data(name=args.data, size=size)
    data = cvt(torch.from_numpy(data))
    return data


data = get_data(args.data_size)
dataset = torch.utils.data.TensorDataset(data)
data_loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=args.batch_size, shuffle=False
)

val_data = get_data(args.data_size)
val_dataset = torch.utils.data.TensorDataset(val_data)
val_data_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=val_data.shape[0], shuffle=False
)


# Model
# --------------------------------------------------------------------------------------
d = 2
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

optimizer = None
if args.optim == "adam":
    optimizer = torch.optim.Adam(
        nfm.parameters(),
        lr=args.lr,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1.00e-08,
        amsgrad=True,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=None,
    )
elif args.optim == "adabelief":
    from adabelief_pytorch import AdaBelief

    optimizer = AdaBelief(
        nfm.parameters(),
        lr=args.lr,
        eps=1.00e-16,
        betas=(0.9, 0.999),
        weight_decouple=True,
        rectify=False,
    )

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=args.lr_drop,
    patience=args.lr_patience,
    threshold=0.001,
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


def save_figure(filename, step=0, iteration=None, ext="png", verbose=True, **kwargs):
    filename = f"{filename}_{step:02.0f}"
    if iteration is not None:
        filename = f"{filename}_{iteration:05.0f}"
    filename = man.get_filename(filename)
    filename = f"{filename}.{ext}"
    kwargs.setdefault("dpi", 250)
    if verbose:
        print(f"Saving file {filename}")
    plt.savefig(filename, **kwargs)
    plt.close()


# Training loop
# --------------------------------------------------------------------------------------


def pushforward_data(data_loader=None, nfm=None, nt=None):
    x_new = torch.zeros(1)
    with torch.no_grad():
        for (x,) in data_loader:
            x_temp, _, _, _ = nfm.forward(x, nt=nt)
            if x_new.shape[0] == 1:
                x_new = x_temp
            else:
                x_new = torch.cat((x_new, x_temp), dim=0)
    dataset = torch.utils.data.TensorDataset(x_new)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=data_loader.batch_size,
        shuffle=False,
    )
    return data_loader


state_dicts = []

for step in range(args.n_steps):
    print(f"step={step}")

    best_avg_loss = best_avg_vloss = float("inf")
    best_state_dict = nfm.state_dict()

    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr
    lr_scheduler.num_bad_epochs = 0

    monitor = Monitor(filename=man.get_filename(f"history_{step:02.0f}.dat"))
    nfm.to(device)
    nfm.train()
    iteration = 0

    for epoch in range(args.n_epochs):
        running_loss = 0.0
        for batch_index, (x,) in enumerate(data_loader):
            x = cvt(x)
            optimizer.zero_grad()
            loss, costs = nfm.forward_kld(x, nt=args.nt, return_costs=True)
            loss.backward()
            optimizer.step()
            monitor.action(
                loss=loss.item(), 
                loss_L=costs[0].item(), 
                loss_C=costs[1].item(), 
                loss_R=costs[2].item(), 
                lr=optimizer.param_groups[0]["lr"],    
            )            
            running_loss += loss
            iteration += 1
        avg_loss = running_loss / (batch_index + 1)

        nfm.eval()
        with torch.no_grad():
            running_vloss = 0.0
            for batch_index, (x,) in enumerate(val_data_loader):
                x = cvt(x)
                vloss = nfm.forward_kld(x, nt=args.val_nt)
                running_vloss += vloss
        avg_vloss = running_vloss / (batch_index + 1)

        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
        if avg_vloss < best_avg_vloss:
            best_avg_vloss = avg_vloss
            best_state_dict = nfm.state_dict()

        lr_scheduler.step(avg_loss)

        print(
            "Epoch {}: avg_loss={:.02e} avg_vloss={:.02e}".format(
                epoch,
                avg_loss,
                avg_vloss,
            )
        )

        if (epoch % args.vis_freq == 0) or (epoch == args.n_epochs - 1):
            with torch.no_grad():
                nfm.eval()
                current_state_dict = nfm.state_dict()
                nfm.load_state_dict(best_state_dict)

                # Plot samples.
                x = val_data[: args.vis_n_points, :]
                plotting.plot_samples(
                    x,
                    nfm,
                    nt=args.val_nt,
                    xmax=4.5,
                    bins=args.vis_n_bins,
                    cmap=args.vis_cmap,
                )
                save_figure("fig_samples", step=step, iteration=iteration)

                # Plot loss history.
                history = monitor.get_data()
                fig, ax = pplt.subplots()
                ax.plot(history["iteration"], history["loss"], color="black")
                ax.format(xlabel="Iteration", ylabel="Loss")
                save_figure("fig__loss", step=step)
                for log in [True, False]:
                    fig, ax = pplt.subplots()
                    for key in ["loss_C", "loss_L", "loss_R"]:
                        ax.plot(history["iteration"], history[key], label=key)
                    if log:
                        ax.format(yscale="log")
                    ax.legend(loc="r", ncols=1, framealpha=0.0)
                    ax.format(xlabel="Iteration", ylabel="Loss")
                    name = "fig__loss_terms_log" if log else "fig__loss_terms"
                    save_figure(name, step=step)

                # Plot learning rate history.
                fig, ax = pplt.subplots()
                ax.plot(history["iteration"], history["lr"], color="black")
                ax.format(xlabel="Iteration", ylabel="Learning rate")
                save_figure("fig__lr", step=step)

                nfm.load_state_dict(current_state_dict)
                nfm.train()

    # Flow forward.
    print("Updating target")
    nfm.load_state_dict(best_state_dict)
    data_loader = pushforward_data(data_loader, nfm, args.nt)
    val_data_loader = pushforward_data(val_data_loader, nfm, args.nt)
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
    filename = man.get_filename(f"checkpoint_{step:02.0f}.pth")
    torch.save(state, filename)

    # Evaluate the chained models.
    print("Plotting")
    with torch.no_grad():
        current_state_dict = nfm.state_dict()
        nfm.eval()

        x = get_data(args.vis_n_points)
        plotting.plot_samples_chained(
            x,
            nfm=nfm,
            state_dicts=state_dicts,
            nt=args.val_nt,
            bins=args.vis_n_bins,
            cmap=args.vis_cmap,
        )
        save_figure("fig_samples_chained", step=step)

        nfm.load_state_dict(current_state_dict)
        nfm.train()
