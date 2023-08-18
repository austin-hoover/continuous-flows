import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import pad
from tqdm import tqdm

from .distributions import DiagGaussian
from .potential import OTPotential


def step_rk4(odefun, z, t0, t1):
    """Apply Runge-Kutta 4 integration step.

    Parameters
    ----------
    odefun : callable
        odefun(z, t) gives dz/dt at time t.
    z : tensor, shape (n, d + 4)
        Inputs. The first n columns represent coordinates. The
    t0, t1 : float
        Start/stop time.

    Returns
    -------
    tensor, shape (n, d + 4)
        The features at time t1.
    """
    h = t1 - t0
    z0 = z

    K = h * odefun(z0, t0)
    z = z0 + (1.0 / 6.0) * K

    K = h * odefun(z0 + 0.5 * K, t0 + (h / 2.0))
    z += (2.0 / 6.0) * K

    K = h * odefun(z0 + 0.5 * K, t0 + (h / 2.0))
    z += (2.0 / 6.0) * K

    K = h * odefun(z0 + K, t0 + h)
    z += (1.0 / 6.0) * K

    return z


class OTFlow(torch.nn.Module):
    """Continuous normalizing flow (CNF) with optimal transport (OT) penalty.

    The forward flow is directed from data space to latent space.

    Attributes
    ----------
    potential : OTPotential
        Neural network representing the OT potential function.
    """
    def __init__(
        self, 
        d=2, 
        m=16, 
        n_layers=2, 
        alpha=[1.0, 1.0, 1.0], 
        base_dist=None, 
    ):
        """
        Parameters
        ----------
        d : int
            Number of input dimensions.
        n_layers : int
            Number of layers.
        m : int
            Number of hidden dimensions.
        alpha : list[float], shape (3,)
            Scaling factors for each term in the cost function (L, C, R).
        base_dist : object
            The base distribution. Must implement `base_dist.sample(n)` 
            and `base_dist.log_prob(x)`. Defaults to unit Gaussian.
        """
        super().__init__()
        self.d = d
        self.m = m
        self.n_layers = n_layers
        self.alpha = alpha
        self.base_dist = base_dist
        if self.base_dist is None:
            self.base_dist = DiagGaussian(d)
        self.potential = OTPotential(n_layers=n_layers, m=m, d=d)

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def odefun(self, x, t):
        """Neural ordinary differential equation (ODE) function.

        Combines the characteristics and log-determinant (Eq. (2)), transport costs
        (Eq. (5)), and HJB regularizer (see Eq. (7)).

        Parameters
        ----------
        z : tensor, shape (n, d)
            Input tensor z = [x; l; v; r].
            - x = particle coordinates, shape (n, d)
            - l = log determinant of the Jacobian, shape (n, 1)
            - v = accumulated transport cost, shape (n, 1)
            - r = accumulated HJB violation, shape (n, 1)
        t : float
            Time coordinate.

        Returns
        -------
        tensor, shape (n, d + 4)
            Output tensor dz/dt.
        """
        nex, d_extra = x.shape
        d = d_extra - 3

        z = pad(x[:, :d], (0, 1, 0, 0), value=t)  # concatenate with the time t

        gradPhi, trH = self.potential.grad_and_hessian_trace(z)

        dx = -(1.0 / self.alpha[0]) * gradPhi[:, 0:d]
        dl = -(1.0 / self.alpha[0]) * trH.unsqueeze(1)
        dv = 0.5 * torch.sum(torch.pow(dx, 2), 1, keepdims=True)
        dr = torch.abs(-gradPhi[:, -1].unsqueeze(1) + self.alpha[0] * dv)

        return torch.cat((dx, dl, dv, dr), 1)

    def step(self, z, t0, t1):
        return step_rk4(self.odefun, z, t0, t1)

    def integrate(self, x, tspan=[0.0, 1.0], nt=8, intermediates=False):
        """Perform the time integration in the d-dimensional space.

        Parameters
        ----------
        x : tensor, shape (n, d)
            Input coordinates.
        tspan : list[float], shape (2,)
            Integration time range; ex: [0.0 , 1.0].
            If tspan=[0, 1], flow from data space to latent space.
            If tspan=[1, 0], flow from latent space to data space.
        nt : int
            The number of time steps.
        intermediates : bool
            If True, save all intermediate time points along the trajectories.

        Returns
        -------
        z : tensor, shape (n, d + 4)
            Output tensor (coordinates, log_det, transport cost, HJB violation).
            Returned if intermediates=False.
        z_full : tensor, shape (n, d + 4, nt + 1)
            Output tensor (coordinates, log_det, transport cost, HJB violation) at each time step.
            Returned if intermediates=True.
        """
        h = (tspan[1] - tspan[0]) / nt

        value = 0.0  # OT-Flow hasvalue = tspan[0]
        z = pad(x, (0, 3, 0, 0), value=value)
        
        tk = tspan[0]
        if intermediates: 
            z_full = torch.zeros(*z.shape, nt + 1, device=x.device, dtype=x.dtype)
            z_full[:, :, 0] = z
            for k in range(nt):
                z_full[:, :, k + 1] = self.step(z_full[:, :, k], tk, tk + h)
                tk += h
            if tspan[0] > tspan[1]:
                z_full[:, -2:, :] *= -1.0  # positive transport costs
            return z_full
        else:
            for k in range(nt):
                z = self.step(z, tk, tk + h)
                tk += h
            if tspan[0] > tspan[1]:
                z[:, -2:] *= -1.0  # positive transport costs
            return z
        return -1

    def unpack(self, z):
        """Return (x, l, v, r) from z."""
        return (
            z[:, : self.d],
            z[:, self.d],
            z[:, self.d + 1],
            z[:, self.d + 2],
        )

    def forward(self, x, nt=8):
        """Flow from data space to latent space.

        Parameters
        ----------
        x : tensor, shape (n, d)
            Input coordinates.

        Returns
        -------
        tensor, shape (n, d)
            Normalized coordinates.
        tensor, shape (n, 1)
            Log determinant of the transformation.
        tensor, shpae (n, 1)
            Accumulated transport cost.
        tensor, shpae (n, 1)
            Accumulated HJB violation cost.
        """
        return self.unpack(self.integrate(x, tspan=[0.0, 1.0], nt=nt))

    def inverse(self, x, nt=8):
        """Flow from latent space to data space.

        Parameters
        ----------
        x : tensor, shape (n, d)
            Input latent-space coordinates.

        Returns
        -------
        tensor, shape (n, d)
            Transformed coordinates.
        tensor, shape (n, 1)
            Log determinant of the transformation.
        tensor, shape (n, 1)
            Accumulated transport cost.
        tensor, shape (n, 1)
            Accumulated HJB violation cost.
        """
        return self.unpack(self.integrate(x, tspan=[1.0, 0.0], nt=nt))

    def forward_kld(self, x, nt=8, return_costs=False):
        """Evaluate the forward KLD + transport costs (see Eq. (8)).

        Samples are transformed to latent space; the log-likelihood is evaluated
        with respect to the base distribution.

        Parameters
        ----------
        x : tensor, shape (n, d)
            Coordinates sampled from the target distribution.
        nt : int
            The number of time steps.
        return_costs : bool
            Whether to return the three costs (L, C, R).

        Returns
        -------
        loss : float
            The objective function value ([alpha_L, alpha_C, alpha_R] \dot [L, C, R]).
        costs : list[float], shape (3,)
            The three computed costs: [L, C, R]. Only returned if `return_costs` is True.
        """
        xn, log_det, L, R = self.forward(x, nt=nt)
        log_prob = self.base_dist.log_prob(xn) + log_det
        cost_L = torch.mean(L)
        cost_C = torch.mean(-log_prob)
        cost_R = torch.mean(R)
        costs = (cost_L, cost_C, cost_R)
        loss = sum(scale * cost for scale, cost in zip(self.alpha, costs))
        if return_costs:
            return (loss, costs)
        else:
            return loss

    def reverse_kld(self, x, nt=8):
        raise NotImplementedError

    def log_prob(self, x, nt=8, intermediates=False):
        """Evaluate the log-probability at x."""
        if intermediates:
            z = self.integrate(x, tspan=[0.0, 1.0], nt=nt, intermediates=True)
            log_det = z[:, self.d, :]
            log_prob = torch.zeros(log_det.shape)
            for i in range(log_prob.shape[-1]):
                log_prob[:, i] = self.base_dist.log_prob(z[:, :self.d, i]) + log_det[:, i]
            return log_prob
        else:
            xn, log_det, _, _ = self.forward(x, nt=nt)
            return self.base_dist.log_prob(xn) + log_det

    def sample(self, n=10, nt=8, batch_size=None):
        """Draw n samples from the model."""
        if batch_size is None:
            batch_size = n
        if batch_size <= n:
            x, _, _, _ = self.inverse(self.base_dist.sample(n), nt=nt)
            return x
        x = torch.zeros(n, self.d)
        for batch_index in range(int(n / batch_size)):
            lo = batch_index * batch_size
            hi = lo + batch_size
            if hi > n:
                hi = n
                batch_size = n - lo
            x[lo:hi], _, _, _ = self.inverse(self.base_dist.sample(batch_size), nt=nt)
        return x







