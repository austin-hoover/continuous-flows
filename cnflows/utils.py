import numpy as np
import torch


def antideriv_tanh(x):
    return torch.abs(x) + torch.log(1.0 + torch.exp(-2.0 * torch.abs(x)))


def deriv_tanh(x):
    return 1.0 - torch.pow(torch.tanh(x), 2)