import copy
import math

import torch
import torch.nn as nn


def antideriv_tanh(x):
    return torch.abs(x) + torch.log(1.0 + torch.exp(-2.0 * torch.abs(x)))


def deriv_tanh(x):
    return 1.0 - torch.pow(torch.tanh(x), 2)


class ResidualNet(nn.Module):
    """Residual neural network."""
    def __init__(self, d=2, m=16, n_layers=2):
        """
        Parameters
        ----------
        d : int
            Dimension of input space (expect inputs to be d + 1 for space-time).
        m : int
            Number of hidden dimensions.
        n_layers : int
            Number of resnet layers (number of theta layers).
        """
        super().__init__()

        if n_layers < 2:
            print("n_layers must be an integer >= 2")
            exit(1)

        self.d = d
        self.m = m
        self.n_layers = n_layers
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(d + 1, m, bias=True))  # opening layer
        self.layers.append(nn.Linear(m, m, bias=True))  # resnet layers
        for i in range(n_layers - 2):
            self.layers.append(copy.deepcopy(self.layers[1]))
        self.act = antideriv_tanh
        self.h = 1.0 / (self.n_layers - 1)  # step size for the ResNet

    def forward(self, x):
        """Forward pass through the network.

        Parameters
        ----------
        x : tensor, shape (nex, d + 1)
            Input tensor.

        Returns
        -------
        tensor, shape (n, m)
            Outputs
        """
        x = self.act(self.layers[0].forward(x))
        for i in range(1, self.n_layers):
            x = x + self.h * self.act(self.layers[i](x))
        return x
        