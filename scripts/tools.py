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


def get_torch_device(gpu=False):
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
    def __init__(self, outdir=None, path=None, use_prefix=True):
        self.outdir = outdir
        self.path = path
        self.datestamp = time.strftime("%Y-%m-%d")
        self.timestamp = time.strftime("%y%m%d%H%M%S")
        self.prefix = f"{self.timestamp}"
        self.outdir = os.path.join(self.outdir, self.timestamp)
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
        self.use_prefix = use_prefix
        print("Output directory: {}".format(self.outdir))
        print("Output file prefix: {}-".format(self.prefix))

    def get_filename(self, filename, sep="-"):
        if not self.use_prefix:
            return os.path.join(self.outdir, filename)
        return os.path.join(self.outdir, "{}{}{}".format(self.prefix, sep, filename))

    def save_script_copy(self):
        shutil.copy(self.path.absolute().as_posix(), self.get_filename(".py", sep=""))

    def get_info(self):
        info = {
            "git_hash": None,
            "git_url": None,
            "outdir": self.outdir,
            "timestamp": self.timestamp,
            "datestamp": self.datestamp,
        }
        return info

    def save_info(self, filename="info.txt"):
        file = open(self.get_filename(filename), "w")
        for key, val in self.get_info().items():
            file.write(f"{key}: {val}\n")
        file.close()

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
    def __init__(self, outdir=None, prefix=None, filename="history.dat", freq=1):
        self.outdir = outdir
        self.prefix = prefix
        self.freq = freq
        self.iteration = 0
        self.history = {
            "epoch": None,
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
        if self.prefix:
            self.filename = f"{self.prefix}-{self.filename}"
        self.filename = os.path.join(self.outdir, self.filename)
        self.file = open(self.filename, mode="w")
        self.file.write(" ".join(list(self.history)))
        self.file.write("\n")

    def print_line(self):
        message = "{:05.0f} epoch={:02.0f}, lr={:0.2e} loss={:9.3e} L={:9.3e} C={:9.3e} R={:9.3e}".format(
            self.history["iteration"],
            self.history["epoch"],
            self.history["lr"],
            self.history["loss"],
            self.history["loss_L"],
            self.history["loss_C"],
            self.history["loss_R"],
        )
        print(message)
        return message

    def write_line(self):
        self.file.write(" ".join([f"{self.history[key]}" for key in self.history]))
        self.file.write("\n")

    def action(self, **kwargs):
        message = None
        if self.iteration % self.freq == 0:
            self.history["iteration"] = self.iteration
            for key, value in kwargs.items():
                if key in self.history:
                    self.history[key] = value
            message = self.print_line()
            self.write_line()
        self.iteration += 1
        return message

    def get_data(self):
        self.file.close()
        data = pd.read_table(self.filename, sep=" ")
        for key in data.columns:
            data[key] = np.array(
                [(float(item) if item != "None" else np.nan) for item in data[key]]
            )
        self.file = open(self.filename, mode="a")
        return data
        