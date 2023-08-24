import logging
import os
import pathlib
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import proplot as pplt
import torch


def get_torch_device_mps(gpu=False):
    """Return torch device on M1 Mac."""
    if gpu:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            if not torch.backends.mps.is_built():
                print(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
    return torch.device("cpu")


class ScriptManager:
    """Helps save script info/output."""
    def __init__(self, outdir=None, path=None, prefix=None):
        self.datestamp = time.strftime("%Y-%m-%d")
        self.timestamp = time.strftime("%y%m%d%H%M%S")
        self.path = path
        folder = self.timestamp
        if prefix is not None:
            folder = f"{prefix}-{folder}"
        self.outdir = os.path.join(outdir, folder)
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
        print("Output directory: {}".format(self.outdir))

    def get_filename(self, filename):
        return os.path.join(self.outdir, filename)

    def save_script_copy(self):
        filename = self.path.absolute().as_posix()
        filename_short = filename.split("/")[-1]
        shutil.copy(filename, self.get_filename(filename_short))

    def get_info(self):
        info = {
            "git_hash": None,
            "git_url": None,
            "outdir": self.outdir,
            "timestamp": self.timestamp,
            "datestamp": self.datestamp,
        }
        return info

    def get_logger(self, save=True, print=True, filename="log.txt"):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if save:
            filename = self.get_filename(filename)
            info_file_handler = logging.FileHandler(filename, mode="a")
            info_file_handler.setLevel(logging.INFO)
            logger.addHandler(info_file_handler)
        if print:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)
        return logger


class Monitor:
    """Helps monitor training."""
    def __init__(self, filename="history.dat", freq=1, memory=10):
        self.history = {
            "iteration": None,
            "lr": None,
            "loss": None,
            "loss_C": None,
            "loss_L": None,
            "loss_R": None,
        }
        self.filename = filename
        if self.filename is None:
            self.filename = "history.dat"
        self.file = open(self.filename, mode="w")
        self.file.write(" ".join(list(self.history)))
        self.file.write("\n")
        
        self.freq = freq
        self.loss_meter = AverageMeter(memory=memory)
        self.iteration = 0
        self.best_loss = float("inf")
        self.best_avg_loss = float("inf")
        self.best_state_dict = None

    def action(
        self, 
        loss=None, 
        loss_L=None, 
        loss_C=None, 
        loss_R=None, 
        lr=None,
    ):
        self.loss_meter.action(loss)
        if self.iteration % self.freq == 0:
            avg_loss = self.loss_meter.average
            if loss < self.best_loss:
                self.best_loss = loss
            if avg_loss < self.best_avg_loss:
                self.best_avg_loss = avg_loss
            print(
                "iter={:05.0f} lr={:0.3e} loss={:0.3e} L={:0.3e} C={:0.3e} R={:0.3e} loss_avg={:0.3e}".format(
                    self.iteration,
                    lr,
                    loss,
                    loss_L,
                    loss_C,
                    loss_R,
                    self.loss_meter.average,
                )
            )
            self.history["iteration"] = self.iteration
            self.history["lr"] = lr
            self.history["loss"] = loss
            self.history["loss_L"] = loss_L
            self.history["loss_C"] = loss_C
            self.history["loss_R"] = loss_R
            self.file.write(" ".join([f"{self.history[key]}" for key in self.history]))
            self.file.write("\n")
        self.iteration += 1

    def get_data(self):
        self.file.close()
        data = pd.read_table(self.filename, sep=" ")
        for key in data.columns:
            data[key] = np.array(
                [(float(item) if item != "None" else np.nan) for item in data[key]]
            )
        self.file = open(self.filename, mode="a")
        return data


class AverageMeter():
    def __init__(self, memory=10):
        self.memory = memory
        self.reset()

    def reset(self):
        self.values = []
        self.average = 0.0

    def action(self, value):
        self.values.append(value)
        self.values = self.values[-self.memory:]
        self.average = np.mean(self.values)
        