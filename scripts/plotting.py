import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cnflows as cnf
import matplotlib.pyplot as plt
import numpy as np
import proplot as pplt
import torch

pplt.rc["grid"] = False
pplt.rc["cycle"] = "538"
pplt.rc["cmap.discrete"] = False
pplt.rc["cmap.sequential"] = "viridis"


def plot_samples(X_true=None, nfm=None, nt=8, bins=50, xmax=4.5, **kwargs):
    """Propagate target samples forward in time and base samples backward in time."""
    kwargs.setdefault("ec", "None")
    kwargs.setdefault("linewidth", 0.0)
    kwargs.setdefault("rasterized", True)
    kwargs.setdefault("shading", "auto")

    limits = [(-xmax, xmax), (-xmax, xmax)]

    X = X_true
    Xn, _, _, _ = nfm.forward(X, nt=nt)
    Yn = nfm.base_dist.sample(X.shape[0])
    Y, _, _, _ = nfm.inverse(Yn, nt=nt)

    fig, axs = pplt.subplots(ncols=2, nrows=2, xspineloc="neither", yspineloc="neither")
    for ax, points in zip(axs, [X, Xn, Y, Yn]):  
        hist, edges = np.histogramdd(
            points.detach().cpu().numpy(), bins, limits, density=True
        )
        ax.pcolormesh(edges[0], edges[1], hist.T, **kwargs)
    axs[0, 0].format(title=r"$x$ ~ $\rho_0$")
    axs[0, 1].format(title=r"$F(x)$")
    axs[1, 0].format(title=r"$F^{-1}(y)$")
    axs[1, 1].format(title=r"$y$ ~ $\rho_1$")
    return axs


def plot_samples_chained(X_true=None, nfm=None, state_dicts=None, nt=8, bins=50, xmax=4.5, **kwargs):
    """Evaluate chained models."""
    kwargs.setdefault("ec", "None")
    kwargs.setdefault("linewidth", 0.0)
    kwargs.setdefault("rasterized", True)
    kwargs.setdefault("shading", "auto")

    limits = [(-xmax, xmax), (-xmax, xmax)]

    X = X_true
    Xn = torch.clone(X)
    for state_dict in state_dicts:
        nfm.load_state_dict(state_dict)
        Xn, _, _, _ = nfm.forward(Xn, nt=nt)

    Yn = cvt(nfm.base_dist.sample(X_true.shape[0]))
    Y = torch.clone(Yn)
    for state_dict in reversed(state_dicts):
        nfm.load_state_dict(state_dict)
        Y, _, _, _ = nfm.inverse(Y, nt=nt)
    
    fig, axs = pplt.subplots(ncols=2, nrows=2, xspineloc="neither", yspineloc="neither")
    for ax, points in zip(axs, [X, Xn, Y, Yn]):  
        hist, edges = np.histogramdd(
            points.detach().cpu().numpy(), bins, limits, density=True
        )
        ax.pcolormesh(edges[0], edges[1], hist.T, **kwargs)
    axs[0, 0].format(title=r"$x$ ~ $\rho_0$")
    axs[0, 1].format(title=r"$F(x)$")
    axs[1, 0].format(title=r"$F^{-1}(y)$")
    axs[1, 1].format(title=r"$y$ ~ $\rho_1$")
    return axs


def plot_loss_history(history, log=False, step=0, **kwargs):
    keys = ["loss", "loss_L", "loss_C", "loss_R"]
    colors = ["black", None, None, None]
    fig, ax = pplt.subplots()
    for key, color in zip(keys, colors):
        ax.plot(history["iteration"], history[key], color=color, label=key, **kwargs)
    ax.legend(ncols=1, loc="right")
    ax.format(xlabel="Iteration", ylabel="Loss")
    if log:
        ax.format(yscale="log")
    return ax
