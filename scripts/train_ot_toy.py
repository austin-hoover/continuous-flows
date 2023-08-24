"""Train OT-Flow on two-dimensional toy data."""
import argparse
import copy
import datetime
import logging
import math
import os
import pathlib
from pprint import pprint
import sys
import time

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

parser.add_argument("--m", type=int, default=16)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--alpha_L", type=float, default=1.0)
parser.add_argument("--alpha_C", type=float, default=100.0)
parser.add_argument("--alpha_R", type=float, default=1.0)
parser.add_argument("--nt", type=int, default=8)

parser.add_argument(
    "--data", 
    type=str, 
    default="circles",
    choices = [
        "2spirals",
        "8gaussians",
        "checkerboard",
        "circles",
        "moons",
        "pinwheel",
        "rings",
        "swissroll",
    ]
)
parser.add_argument("--batch_size", type=int, default=2000)

parser.add_argument("--n_iters", type=int, default=2000)
parser.add_argument("--lr", type=float, default=1.00e-02)
parser.add_argument("--min_lr", type=float, default=1.00e-03)
parser.add_argument("--lr_drop", type=float, default=0.5)
parser.add_argument("--lr_patience", type=int, default=200)
parser.add_argument("--lr_thresh", type=float, default=0.0001)
parser.add_argument("--sample_freq", type=int, default=20)

parser.add_argument("--optim", type=str, default="adam", choices=["adam", "adabelief"])

parser.add_argument("--vis_freq", type=int, default=100)
parser.add_argument("--vis_n_points", type=int, default=int(10.00e+04))
parser.add_argument("--vis_n_bins", type=int, default=100)
parser.add_argument("--vis_cmap", type=str, default="viridis")

parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--outdir", type=str, default="./data_output/")
parser.add_argument("--precision", type=str, default="single", choices=["single", "double"])
parser.add_argument("--print_freq", type=int, default=1)

args = parser.parse_args()


# Settings
# --------------------------------------------------------------------------------------
device = get_torch_device_mps(gpu=args.gpu)

precision = torch.float32
if args.precision == "double":
    precision = torch.float64
torch.set_default_dtype(precision)
cvt = lambda x: x.type(precision).to(device, non_blocking=True)

man = ScriptManager(outdir=args.outdir, path=pathlib.Path(__file__), use_prefix=False)
man.save_script_copy()
pprint(man.get_info())

logger = man.get_logger(save=True, print=True, filename="log.txt")
for key, val in man.get_info().items():
    logger.info(f"{key}: {val}")
logger.info(args)


# Data
# --------------------------------------------------------------------------------------
data = cnf.utils.gen_toy_data(name=args.data, size=args.batch_size)
data = cvt(torch.from_numpy(data))


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
logger.info("------------------------------------------------------------")
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
        amsgrad=False, 
        foreach=None, 
        maximize=False, 
        capturable=False, 
        differentiable=False, 
        fused=None
    )
elif args.optim == "adabelief":
    from adabelief_pytorch import AdaBelief
    optimizer = AdaBelief(
        nfm.parameters(), 
        lr=args.lr, 
        weight_decouple=False, 
        weight_decay=0.0,
        betas=(0.9,0.999), 
        eps=1.00e-16, 
        rectify=False,
    )

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=args.lr_drop,
    patience=args.lr_patience,
    threshold=args.lr_thresh,
    threshold_mode="rel",
    min_lr=args.min_lr,
    verbose=False,
)

logger.info("")
logger.info("Optimizer")
logger.info("------------------------------------------------------------")
logger.info(optimizer)


# Training loop
# --------------------------------------------------------------------------------------
monitor = Monitor(outdir=man.outdir, filename=f"history.dat", print=False)
best_loss = float("inf")
best_state_dict = nfm.state_dict()
nfm.train()
for iteration in range(args.n_iters):
    optimizer.zero_grad()
    loss, costs = nfm.forward_kld(data, nt=args.nt, return_costs=True)
    loss.backward()
    optimizer.step()
    lr_scheduler.step(loss)
    monitor.action(
        loss=loss, 
        lr=optimizer.param_groups[0]["lr"], 
        loss_L=costs[0],
        loss_C=costs[1], 
        loss_R=costs[2]
    )
    print(
        "{:05.0f} lr={:0.2e} loss={:9.3e} L={:9.3e} C={:9.3e} R={:9.3e} n_bad={}".format(
            iteration,
            optimizer.param_groups[0]["lr"],
            loss,
            costs[0],
            costs[1],
            costs[2],
            lr_scheduler.num_bad_epochs,
        )
    )

    if iteration % args.vis_freq == 0 or iteration == args.n_iters - 1:
        with torch.no_grad():
            nfm.eval()
            current_state_dict = nfm.state_dict()
            nfm.load_state_dict(best_state_dict)

            # Plot samples
            x = cnf.utils.gen_toy_data(name=args.data, size=args.vis_n_points)
            x = cvt(torch.from_numpy(x))
            plotting.plot_samples(
                x,
                nfm,
                nt=args.nt,
                xmax=4.5,
                bins=args.vis_n_bins,
                cmap=args.vis_cmap,
            )
            filename = man.get_filename(f"fig_samples_{iteration:05.0f}.png")
            print("Saving file", filename)
            plt.savefig(filename, dpi=200)
            plt.close()

            # Plot loss history.
            history = monitor.get_data()
            fig, ax = pplt.subplots()
            ax.plot(history["iteration"], history["loss"], color="black")
            ax.format(xlabel="Iteration", ylabel="Loss")
            filename = man.get_filename(f"fig_loss.png")
            print("Saving file", filename)
            plt.savefig(filename, dpi=200)
            plt.close()

            # Plot costs history.
            history = monitor.get_data()
            fig, ax = pplt.subplots()
            for key in ["loss_C", "loss_L", "loss_R"]:
                ax.plot(history["iteration"], history[key], label=key)
            ax.format(xlabel="Iteration", ylabel="Loss")
            ax.legend(loc="r", ncols=1, framealpha=0.0)
            filename = man.get_filename(f"fig_loss_terms.png")
            print("Saving file", filename)
            plt.savefig(filename, dpi=200)
            plt.close()

            # Plot learning rate history.
            fig, ax = pplt.subplots()
            ax.plot(history["iteration"], history["lr"], color="black")
            ax.format(xlabel="Iteration", ylabel="Learning rate")
            filename = man.get_filename(f"fig_lr.png")
            print("Saving file", filename)
            plt.savefig(filename, dpi=200)
            plt.close()

            nfm.load_state_dict(current_state_dict)
            nfm.train()

    if iteration % args.sample_freq == 0:
        print("Resampling")
        data = cnf.utils.gen_toy_data(name=args.data, size=args.batch_size)
        data = cvt(torch.from_numpy(data))

