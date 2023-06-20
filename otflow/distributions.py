"""
Source: https://github.com/VincentStimper/normalizing-flows/tree/master/normflows/distributions.
"""
import math
import numpy as np
import scipy.special
import torch


class Distribution(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def log_prob(self, z):
        raise NotImplementedError

    def sample(self, n=1, **kws):
        raise NotImplementedError


class Gaussian(Distribution):
    def __init__(self, d=2):
        super().__init__()
        self.d = d
        self.loc = torch.nn.Parameter(torch.zeros(1, self.d))
        self.log_scale = torch.nn.Parameter(torch.zeros(1, self.d))

    def log_prob(self, z):
        loc = self.loc
        log_scale = self.log_scale
        log_p = -0.5 * self.d * math.log(2.0 * math.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - loc) / torch.exp(log_scale), 2),
            1,
        )
        return log_p

    def sample(self, n=1):
        return torch.randn((n, self.d))


class Uniform(Distribution):
    def __init__(self, d=2, low=-1.0, high=1.0):
        super().__init__()
        self.d = d
        self.low = torch.tensor(low)
        self.high = torch.tensor(high)
        self.log_prob_val = -self.d * np.log(self.high - self.low)

    def log_prob(self, z):
        log_p = self.log_prob_val * torch.ones(z.shape[0], device=z.device)
        out_range = torch.logical_or(z < self.low, z > self.high)
        ind_inf = torch.any(torch.reshape(out_range, (z.shape[0], -1)), dim=-1)
        log_p[ind_inf] = -np.inf
        return log_p

    def sample(self, n=10):
        return 2.0 * torch.rand((n, self.d)) - 1.0
        

class Waterbag(Distribution):
    def __init__(self, d=2):
        self.d = d
        
    def log_prob(self, z):
        C = (np.pi ** (0.5 * self.d)) / scipy.special.gamma(1.0 + 0.5 * self.d)
        radii = torch.sqrt(torch.sum(torch.square(z), dim=1))
        _log_prob = torch.full((z.shape[0],), np.log(1.0 / C))
        _log_prob[radii > 1.0] = -np.inf
        return _log_prob

    def sample(self, n=10):
        z = torch.randn((n, self.d))  # normal
        z = z / torch.sqrt(torch.sum(torch.square(z), dim=1))[:, None]  # surface of unit ball
        r = torch.rand((n, 1)) ** (1.0 / self.d)
        return r * z


class TwoMoons(Distribution):
    def __init__(self):
        super().__init__()
        self.d = 2
        self.max_log_prob = 0.0

    def log_prob(self, z):
        a = torch.abs(z[:, 0])
        log_prob = (
            -0.5 * ((torch.norm(z, dim=1) - 2) / 0.2) ** 2
            - 0.5 * ((a - 2) / 0.3) ** 2
            + torch.log(1 + torch.exp(-4 * a / 0.09))
        )
        return log_prob
