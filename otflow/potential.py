"""Neural network model of the OT potential function."""
import copy
import math
import time

import torch
import torch.nn as nn


def anti_deriv_tanh(x):
    """Activation function aka the anti_derivative of tanh."""
    return torch.log(torch.exp(x) + torch.exp(-x))


def deriv_tanh(x):
    """act'' aka the second derivative of the activation function anti_deriv_tanh."""
    return 1 - torch.pow(torch.tanh(x), 2)


class ResNN(nn.Module):
    """Residual neural network."""

    def __init__(self, d=2, m=32, nTh=2):
        """
        Parameters
        ----------
        d : int
            Dimension of space input (expect inputs to be d + 1 for space-time).
        m : int
            Hidden dimension.
        nTh : int
            Number of ResNet layers.
        """
        super().__init__()

        if nTh < 2:
            raise ValueError("nTh mus be an integer >= 2.")

        self.d = d
        self.m = m
        self.nTh = nTh
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(d + 1, m, bias=True))  # opening layer
        self.layers.append(nn.Linear(m, m, bias=True))  # resnet layers
        for i in range(nTh - 2):
            self.layers.append(copy.deepcopy(self.layers[1]))
        self.act = anti_deriv_tanh
        self.h = 1.0 / (self.nTh - 1.0)  # step size for the ResNet

    def forward(self, x):
        """The forward propogation of the ResNet N(s; theta).

        Parameters
        ----------
        x : tensor, shape (nex, d + 1)

        Returns
        -------
        torch.Tensor, shape (nex, m)
        """
        x = self.act(self.layers[0].forward(x))
        for i in range(1, self.nTh):
            x = x + self.h * self.act(self.layers[i](x))
        return x


class Potential(nn.Module):
    """Neural network approximating the potential Phi.
    
    Phi(x, t) = w' * ResNet([x; t]) + 0.5 * [x' t] * A'A * [x; t] + b'*[x; t] + c
    """
    def __init__(self, nTh=None, m=None, d=None, r=10, alpha=[1.0, 1.0, 1.0, 1.0, 1.0]):
        """
        Parameters
        ----------
        nTh : int
            The number of resNet layers.
        m : int
            The number of hidden dimensions.
        d : int
            The dimension of space input (expect inputs to be d+1 for space-time).
        r : int
            The rank of the A matrix.
        alpha : list
            The alpha values / weighted multipliers for the optimization problem.
        """
        super().__init__()

        self.m = m
        self.nTh = nTh
        self.d = d
        self.alpha = alpha

        # If the number of dimensions is smaller than the default r, use that.
        r = min(r, d + 1)

        self.A = nn.Parameter(torch.zeros(r, d + 1), requires_grad=True)
        self.A = nn.init.xavier_uniform_(self.A)
        self.c = nn.Linear(d + 1, 1, bias=True)  # b' * [x; t] + c
        self.w = nn.Linear(m, 1, bias=False)

        self.N = ResNN(d, m, nTh=nTh)

        # Set initial values.
        self.w.weight.data = torch.ones(self.w.weight.data.shape)
        self.c.weight.data = torch.zeros(self.c.weight.data.shape)
        self.c.bias.data   = torch.zeros(self.c.bias.data.shape)

    def forward(self, x):
        """Calculate Phi(s, theta). (Not used in OT-Flow)."""
        symA = torch.matmul(torch.t(self.A), self.A)  # A'A
        return self.w(self.N(x)) + 0.5 * torch.sum(torch.matmul(x, symA) * x, dim=1, keepdims=True) + self.c(x)

    def hessian_trace(self, x, just_grad=False):
        """Compute gradient of Phi wrt x and trace(Hessian of Phi). 
        
        See Eq. (11) and Eq. (13), respectively.

        Recomputes the forward propogation portions of Phi.

        Parameters
        ----------
        x : torch.Tensor, nex-by-d
            Input data.
        just_grad : bool
            If True, only return gradient; if False return (grad, trHess).

        Returns
        -------
        gradient :
            The gradient of Phi wrt x.
        trace(hessian) :
            Trace of Hessian of Phi.
        """
        # Code in E = eye(d + 1, d) as index slicing instead of matrix multiplication
        # assumes specific N.act as the anti_derivative of tanh.

        N = self.N
        m = N.layers[0].weight.shape[0]
        nex = x.shape[0] # number of examples in the batch
        d = x.shape[1] - 1
        symA = torch.matmul(self.A.t(), self.A)

        # Hold the u_0, u_1, ..., u_M for the forward pass.
        u = []
        
        # Hold the z_0, z_1, ..., z_M for the backward pass. # Pre-allocate z because we will 
        # store in the backward pass and we want the indices to match the paper.
        z = N.nTh * [None]
        
        # Forward of ResNet N and fill u.
        opening = N.layers[0].forward(x)  # K_0 * S + b_0
        u.append(N.act(opening))  # u0
        feat = u[0]

        for i in range(1, N.nTh):
            feat = feat + N.h * N.act(N.layers[i](feat))
            u.append(feat)

        # Going to be used more than once:
        tanhopen = torch.tanh(opening)  # act'( K_0 * S + b_0 )

        # Compute gradient and fill z. Work backwards, placing z_i in appropriate spot.
        for i in range(N.nTh-1, 0, -1):
            if i == N.nTh - 1:
                term = self.w.weight.t()
            else:
                term = z[i+1]
            ## z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            z[i] = term + N.h * torch.mm(N.layers[i].weight.t(), torch.tanh(N.layers[i].forward(u[i - 1])).t() * term)

        ## z_0 = K_0' diag(...) z_1
        z[0] = torch.mm(N.layers[0].weight.t(), tanhopen.t() * z[1])
        grad = z[0] + torch.mm(symA, x.t()) + self.c.weight.t()

        if just_grad:
            return grad.t()

        # -----------------
        # Hessian trace
        #-----------------

        # t_0, the trace of the opening layer
        Kopen = N.layers[0].weight[:,0:d]   # indexed version of Kopen = torch.mm(N.layers[0].weight, E)
        temp = deriv_tanh(opening.t()) * z[1]
        trH = torch.sum(temp.reshape(m, -1, nex) * torch.pow(Kopen.unsqueeze(2), 2), dim=(0, 1))  # trH = t_0

        # grad_s u_0 ^ T
        temp = tanhopen.t()  # act'( K_0 * S + b_0 )
        Jac  = Kopen.unsqueeze(2) * temp.unsqueeze(1) # K_0' * act'( K_0 * S + b_0 )
        # Jac is shape m by d by nex

        # t_i, trace of the resNet layers
        # KJ is the K_i^T * grad_s u_{i-1}^T
        for i in range(1, N.nTh):
            KJ = torch.mm(N.layers[i].weight, Jac.reshape(m, -1))
            KJ = KJ.reshape(m, -1, nex)
            if i == N.nTh - 1:
                term = self.w.weight.t()
            else:
                term = z[i + 1]

            temp = N.layers[i].forward(u[i - 1]).t()  # (K_i * u_{i-1} + b_i)
            t_i = torch.sum((deriv_tanh(temp) * term).reshape(m, -1, nex) * torch.pow(KJ, 2), dim=(0, 1))
            trH = trH + N.h * t_i  # add t_i to the accumulate trace
            Jac = Jac + N.h * torch.tanh(temp).reshape(m, -1, nex) * KJ  # update Jacobian

        ## Indexed version of: `return (grad.t(), trH + torch.trace(torch.mm(E.t(), torch.mm(symA, E))))`.
        return grad.t(), trH + torch.trace(symA[:d, :d])