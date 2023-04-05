import math

import torch


class Gaussian:
    """Source: https://github.com/EmoryMLIP/OT-Flow/blob/master/src/OTFlowProblem.py"""

    def __init__(self, d=2, trainable=True):
        self.d = d
        if trainable:
            self.loc = torch.nn.Parameter(torch.zeros(1, self.d))
            self.log_scale = torch.nn.Parameter(torch.zeros(1, self.d))
        else:
            self.register_buffer("loc", torch.zeros(1, self.d))
            self.register_buffer("log_scale", torch.zeros(1, self.d))
        self.temperature = None  # Temperature parameter for annealed sampling

    def log_prob(self, z):
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + math.log(self.temperature)
        log_p = -0.5 * self.d * math.log(2.0 * math.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(log_scale), 2),
            1,
        )
        return log_p
    
    def sample(self, n):
        return torch.randn((n, self.d))
