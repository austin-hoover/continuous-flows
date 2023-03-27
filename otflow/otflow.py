"""Optimal transport normalizing flow (OT-Flow)."""
import math

import torch
from torch.nn.functional import pad


def odefun(x, t, potential, alpha=[1.0, 1.0, 1.0]):
    """Neural ordinary differential equation (ODE) function.

    Combines the characteristics and log-determinant (Eq. (2)), transport costs 
    (Eq. (5)), and HJB regularizer (see Eq. (7)).

    d_t [x; l; v; r] = odefun([x; l; v; r] , t)

    x = particle position
    l = log determinant
    v = accumulated transport costs (Lagrangian).
    r = accumulated violation of HJB condition.
    """
    nex, d_extra = x.shape
    d = d_extra - 3
    z = pad(x[:, :d], (0, 1, 0, 0), value=t)  # concatenate with the time t
    grad_Phi, tr_H = potential.hessian_trace(z)
    dx = -(1.0 / alpha[0]) * grad_Phi[:, :d]
    dl = -(1.0 / alpha[0]) * tr_H.unsqueeze(1)
    dv = 0.5 * torch.sum(torch.pow(dx, 2), 1, keepdims=True)
    dr = torch.abs(-grad_Phi[:, -1].unsqueeze(1) + alpha[0] * dv)
    return torch.cat((dx, dl, dv, dr), 1)


def step_RK4(odefun, z, potential, alpha, t0, t1):
    """Runge-Kutta 4 integration method.

    Parameters
    ----------
    odefun : callable
        Function to apply at every time step.
    z : tensor, shape (nex, d + 4)
        Inputs.
    potential : torch.nn.Module
        Neural network representing the potential function.
    alpha : list[float], shape (3,)
        The 3 alpha values for the OT-Flow Problem.
    t0, t1 : float
        Start/stop time.

    Returns
    -------
    tensor, shape (nex, d + 4)
        The features at time t1.
    """
    h = t1 - t0
    z0 = z

    K = h * odefun(z0, t0, potential, alpha=alpha)
    z = z0 + (1.0 / 6.0) * K

    K = h * odefun(z0 + 0.5 * K, t0 + (h / 2), potential, alpha=alpha)
    z += (2.0 / 6.0) * K

    K = h * odefun(z0 + 0.5 * K, t0 + (h / 2), potential, alpha=alpha)
    z += (2.0 / 6.0) * K

    K = h * odefun(z0 + K, t0 + h, potential, alpha=alpha)
    z += (1.0 / 6.0) * K

    return z


def step_RK1(odefun, z, potential, alpha, t0, t1):
    """Runge-Kutta 1 / Forward Euler integration method.  
        
    Parameters
    ----------
    odefun : callable
        Function to apply at every time step.
    z : tensor, shape (nex, d+4)
        Inputs.
    potential : torch.nn.Module
        Neural network representing the potential function.
    alpha : list[float], shape (3,)
        The 3 alpha values for the mean field game problem.
    t0, t1 : float
        Start/stop time.
        
    Returns
    -------
    tensor, shape (nex, d + 4)
        The features at time t1.
    """
    z += (t1 - t0) * odefun(z, t0, potential, alpha=alpha)
    return z


def negative_log_likelihood(z):
    """Expected negative log-likelihood; see Eq.(3) in the paper."""
    d = z.shape[1] - 3
    l = z[:, d].unsqueeze(1)  # log-determinant
    return torch.sum(0.5 * math.log(2.0 * math.pi) + 0.5 * torch.pow(z[:, :d], 2), 1, keepdims=True) - l


def objective(x, potential, tspan=[0.0, 1.0], nt=8, method="rk4", alpha=[1.0, 1.0, 1.0]):
    """Evaluate objective function of the OT-Flow problem.

    See Eq. (8) in the paper.

    Parameters
    ----------
    x : tensor, shape (nex, d)
        Input data tensor.
    potential : torch.nn.Module
        Neural network representing the potential function.
    tspan : list[float], shape (2,)
        Integration time range; ex: [0.0 , 1.0].
    nt : int
        The number of time steps.
    method : {"rk1", "rk4"}
        The integration methods.
    alpha : list[float], shape (3,)
        The alphaa value multipliers for each term in the cost function (L, C, R).

    Returns
    -------
    loss : float
        The objective function value [alpha_L, alpha_C, alpha_R] * [L, C, R].
    costs : list[float], shape (3,)
        The three computed costs: [L, C, R].
    """
    h = (tspan[1] - tspan[0]) / nt
    tk = tspan[0]

    # Initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs.
    z = pad(x, (0, 3, 0, 0), value=0)

    if method == "rk4":
        for k in range(nt):
            z = step_RK4(odefun, z, potential, alpha, tk, tk + h)
            tk += h
    elif method == "rk1":
        for k in range(nt):
            z = step_RK1(odefun, z, potential, alpha, tk, tk + h)
            tk += h

    # Assume all examples are equally weighted.
    cost_L = torch.mean(z[:, -2])
    cost_C = torch.mean(negative_log_likelihood(z))
    cost_R = torch.mean(z[:, -1])

    costs = [cost_L, cost_C, cost_R]
    loss = sum(i[0] * i[1] for i in zip(costs, alpha))
    return loss, costs


def integrate(
    x, network, tspan=[0.0, 1.0], nt=8, method="rk4", alpha=[1.0, 1.0, 1.0], intermediates=False
):
    """Perform the time integration in the d-dimensional space.

    Parameters
    ----------
    x : tensor, shape (nex, d)
        Input data tensor.
    network : torch.nn.Module
        Neural network Phi.
    tspan : list[float], shape (2,)
        Integration time range; ex: [0.0 , 1.0].
    nt : int
        The number of time steps.
    method : {"rk1", "rk4"}
        The integration method.
    alpha : list[float], shape (3,)
        The alpha value multipliers.
    intermediates : bool
        If True, save all intermediate time points along the trajectories.

    Returns
    -------
    z : tensor, shape (nex, d + 4)
        The features at time t1. (Returned if `intermediates=False`.)
    z_full : tensor, shape (nex, d + 3, nt + 1)
        Trajectories from time t0 to t1. (Returned if `intermediates=True`.)
    """
    h = (tspan[1] - tspan[0]) / nt
    tk = tspan[0]

    # Initialize "hidden" vector to propagate with all the additional dimensions for all the ODEs.
    z = pad(x, (0, 3, 0, 0), value=tspan[0])

    if intermediates:
        z_full = torch.zeros(*z.shape, nt + 1, device=x.device, dtype=x.dtype)
        z_full[:, :, 0] = z
        if method == "rk4":
            for k in range(nt):
                z_full[:, :, k + 1] = step_RK4(
                    odefun, z_full[:, :, k], network, alpha, tk, tk + h
                )
                tk += h
        elif method == "rk1":
            for k in range(nt):
                z_full[:, :, k + 1] = step_RK1(
                    odefun, z_full[:, :, k], network, alpha, tk, tk + h
                )
                tk += h

        return z_full
    else:
        if method == "rk4":
            for k in range(nt):
                z = step_RK4(odefun, z, network, alpha, tk, tk + h)
                tk += h
        elif method == "rk1":
            for k in range(nt):
                z = step_RK1(odefun, z, network, alpha, tk, tk + h)
                tk += h
        return z
    return -1