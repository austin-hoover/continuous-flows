"""Optimal transport normalizing flow (OT-Flow)."""
import math

import torch
from torch.nn.functional import pad

from .potential import Potential


class ContinuousNormalizingFlow:
    """Continuous normalizing flow (CNF).
    
    Parameters
    ----------
    base : object
        The base distribution (defined at t=1, the final state of the
        normalizing flow).
    """
    def __init__(self, base=None, d=2):
        self.base = base
        self.d = d
        
    def odefun(self, x, t):
        """Neural ordinary differential equation (ODE) function."""
        raise NotImplementedError()
    
    def step(self, z, t0, t1):
        """Apply Runge-Kutta 4 integration step.

        Parameters
        ----------
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

        K = h * self.odefun(z0, t0)
        z = z0 + (1.0 / 6.0) * K

        K = h * self.odefun(z0 + 0.5 * K, t0 + (h / 2))
        z += (2.0 / 6.0) * K

        K = h * self.odefun(z0 + 0.5 * K, t0 + (h / 2))
        z += (2.0 / 6.0) * K

        K = h * self.odefun(z0 + K, t0 + h)
        z += (1.0 / 6.0) * K

        return z
                
    def integrate(self, x, tspan=[0.0, 1.0], nt=8, intermediates=False):
        """Track tensor through the system."""
        raise NotImplementedError()
        
    def objective(self, x, nt=8):
        """Compute loss."""
        raise NotImplementedError()
                

class OptimalTransport(ContinuousNormalizingFlow):
    """Continuous normalizing flow (CNF) with optimal transport (OT) penalty.

    Attributes
    ----------
    network : torch.nn.Module
        Neural network representing the OT potential function. Can also access
        with the alias `potential`.
    """

    def __init__(self, base=None, alpha=[1.0, 100.0, 5.0], nTh=2, m=32, d=2, precision=torch.float64, device=None):
        """
        Parameters
        ----------
        base : object
            The base distribution.
        alpha : list[float], shape (3,)
            Scaling factors for each term in the cost function (L, C, R).
        nTh : int
            Number of layers.
        m : int
            Number of hidden dimensions.
        d : int
            Number of dimensions.
        """
        super().__init__(base=base)
        self.d = d
        self.alpha = alpha
        self.m = m
        self.nTh = nTh
        self.device = device
        self.network = Potential(nTh=nTh, m=m, d=d, alpha=alpha)
        self.network = self.network.to(precision).to(device)
        self.potential = self.network

    def odefun(self, x, t):
        """Neural ordinary differential equation (ODE) function.

        Combines the characteristics and log-determinant (Eq. (2)), transport costs
        (Eq. (5)), and HJB regularizer (see Eq. (7)).

        d_t [x; l; v; r] = odefun([x; l; v; r] , t)

        x = particle coordinates
        l = log determinant of the Jacobian
        v = accumulated transport cost (Lagrangian)
        r = accumulated violation of HJB condition
        """
        n, d_extra = x.shape
        d = d_extra - 3
        z = pad(x[:, :d], (0, 1, 0, 0), value=t)  # concatenate with the time t
        grad_Phi, tr_H = self.network.hessian_trace(z)
        dx = -(1.0 / self.alpha[0]) * grad_Phi[:, :d]
        dl = -(1.0 / self.alpha[0]) * tr_H.unsqueeze(1)
        dv = 0.5 * torch.sum(torch.pow(dx, 2), 1, keepdims=True)
        dr = torch.abs(-grad_Phi[:, -1].unsqueeze(1) + self.alpha[0] * dv)
        return torch.cat((dx, dl, dv, dr), 1)
    
    def integrate(self, x, tspan=[0.0, 1.0], nt=8, intermediates=False):
        """Perform the time integration in the d-dimensional space.
        
        If tspan=(0, 1), flow from data space to latent space. 
        If tspan=(1, 0), flow from latent space to data space.
        
        Parameters
        ----------
        x : tensor, shape (n, d)
            Input coordinates.
        tspan : list[float], shape (2,)
            Integration time range; ex: [0.0 , 1.0].
        nt : int
            The number of time steps.
        intermediates : bool
            If True, save all intermediate time points along the trajectories.

        Returns
        -------
        z : tensor, shape (n, d + 4)
            The features at time t1. (Returned if `intermediates=False`.)
            Columns are defined above (coordinates, log_det, transport cost, HJB violation).
        z_full : tensor, shape (n, d + 3, nt + 1)
            Trajectories from time t0 to t1. (Returned if `intermediates=True`.)
            Columns are defined above.
        """
        h = (tspan[1] - tspan[0]) / nt
        tk = tspan[0]

        # Initialize "hidden" vector to propagate with all additional dimensions for all ODEs.
        z = pad(x, (0, 3, 0, 0), value=tspan[0])

        if intermediates:
            z_full = torch.zeros(*z.shape, nt + 1, device=x.device, dtype=x.dtype)
            z_full[:, :, 0] = z
            for k in range(nt):
                z_full[:, :, k + 1] = self.step(z_full[:, :, k], tk, tk + h)
                tk += h
            return z_full
        else:
            for k in range(nt):
                z = self.step(z, tk, tk + h)
                tk += h
            return z
                
    def objective(self, x, nt=8, return_costs=False):
        """Evaluate the objective function (Eq. (8)).
        
        Samples are transformed to latent space; the log-likelihood is evaluated 
        with respect to the base distribution.

        Parameters
        ----------
        x : tensor, shape (n, d)
            Coordinates sampled from the target distribution.
        tspan : list[float], shape (2,)
            Integration time range; ex: [0.0 , 1.0].
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
        z = self.integrate(x, tspan=[0.0, 1.0], nt=nt, intermediates=False)
        log_det = z[:, self.d].unsqueeze(1)
        log_prob = self.base.log_prob(z[:, :self.d]) + log_det

        # Assume all examples are equally weighted.
        cost_L = torch.mean(z[:, -2])
        cost_C = torch.mean(-log_prob)
        cost_R = torch.mean(z[:, -1])
        
        costs = [cost_L, cost_C, cost_R]
        loss = sum(i[0] * i[1] for i in zip(costs, self.alpha))
        if return_costs:
            return (loss, costs)
        else:
            return loss
        
    def objective_reverse(self, nt=8, return_costs=False):
        raise NotImplementedError
            
    def unpack(self, x):
        """Return (x, l, v, r)."""
        return x[:, :self.d], x[:, self.d], x[:, self.d + 1], x[:, self.d + 2]
    
    def log_prob(self, x, nt=8, intermediates=False):
        """Evaluate the predicted log-probability."""
        if intermediates:
            z = self.integrate(x, tspan=[0.0, 1.0], nt=nt, intermediates=True)            
            log_det = z[:, self.d, :]
            log_prob = torch.zeros(log_det.shape)
            for i in range(log_prob.shape[-1]):
                log_prob[:, i] = self.base.log_prob(z[:, :self.d, i]) + log_det[:, i]
            return log_prob
        else:
            z, log_det, _, _ = self.unpack(self.integrate(x, tspan=[0.0, 1.0], nt=nt))
            return self.base.log_prob(z) + log_det
        
    def forward(self, z, nt=8):
        """Flow from latent space to data space."""
        return self.upack(self.integrate(z, tspan=[1.0, 0.0], nt=nt))
    
    def inverse(self, x, nt=8):
        """Flow from data space to latent space."""
        return self.upack(self.integrate(x, tspan=[0.0, 1.0], nt=nt))