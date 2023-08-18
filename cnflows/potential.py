import torch
import torch.nn as nn

from .networks import antideriv_tanh
from .networks import deriv_tanh
from .networks import ResidualNet


class OTPotential(nn.Module):
    """Neural network approximating the OT potential Phi (Eq. (9)).

    Phi(x, t) = w' * ResNet([x;t]) + 0.5*[x' t] * A'A * [x;t] + b'*[x;t] + c
    """
    def __init__(self, n_layers=2, m=16, d=2, r=10, alpha=(1.0, 1.0, 1.0, 1.0, 1.0)):
        """Constructor.
        
        Parameters
        ----------
        n_layers : int
            The number of resnet layers (number of theta layers).
        m : int
            Hidden dimension (network width).
        d : int
            Dimension of input space (expect inputs to be d + 1 for space-time).
        r : int
            Rank r for the A matrix.
        """
        super().__init__()

        self.m = m
        self.n_layers = n_layers
        self.d = d
        self.alpha = alpha

        r = min(r, d + 1)  # if number of dimensions is smaller than default r, use that

        self.A = nn.Parameter(torch.zeros(r, d + 1), requires_grad=True)
        self.A = nn.init.xavier_uniform_(self.A)
        self.c = nn.Linear(d + 1, 1, bias=True)  # b'*[x;t] + c
        self.w = nn.Linear(m, 1, bias=False)

        self.N = ResidualNet(d=d, m=m, n_layers=n_layers)

        # set initial values
        self.w.weight.data = torch.ones(self.w.weight.data.shape)
        self.c.weight.data = torch.zeros(self.c.weight.data.shape)
        self.c.bias.data = torch.zeros(self.c.bias.data.shape)

    def forward(self, x):
        """Calculating Phi(s, theta)...not used in OT-Flow."""

        # force A to be symmetric
        symA = torch.matmul(torch.t(self.A), self.A)  # A'A

        return self.w(self.N(x)) + 0.5 * torch.sum(torch.matmul(x, symA) * x, dim=1, keepdims=True) + self.c(x)

    def grad_and_hessian_trace(self, x, just_grad=False):
        """Compute gradient of Phi with respect to x and trace(Hessian(Phi)).

        Seee Eq. (11) and Eq. (13), respectively.

        This function recomputes the forward propogation portions of Phi.

        Parameters
        ----------
        x : tensor, shape (n, d)
            Input data.
        just_grad : bool
            If true, only return gradient; if false, return (gradient, hessian_trace).

        Returns
        -------
        tensor, shape
            The gradient of Phi with respect to x.
        tensor, shape
            The trace of the Hessian of Phi. Returned if just_grad=True.
        """
        # code in E = eye(d+1,d) as index slicing instead of matrix multiplication
        # assumes specific N.act as the antiderivative of tanh

        N = self.N
        m = N.layers[0].weight.shape[0]
        nex = x.shape[0]  # number of examples in the batch
        d = x.shape[1] - 1
        symA = torch.matmul(self.A.t(), self.A)

        u = []  # hold the u_0,u_1,...,u_M for the forward pass
        z = N.n_layers * [None]  # hold the z_0,z_1,...,z_M for the backward pass
        # preallocate z because we will store in the backward pass and we want the indices to match the paper

        # Forward of ResNet N and fill u
        opening = N.layers[0].forward(x)  # K_0 * S + b_0
        u.append(N.act(opening))  # u0
        feat = u[0]

        for i in range(1, N.n_layers):
            feat = feat + N.h * N.act(N.layers[i](feat))
            u.append(feat)

        # going to be used more than once
        tanhopen = torch.tanh(opening)  # act'( K_0 * S + b_0 )

        # compute gradient and fill z
        for i in range(N.n_layers - 1, 0, -1):  # work backwards, placing z_i in appropriate spot
            if i == N.n_layers - 1:
                term = self.w.weight.t()
            else:
                term = z[i + 1]

            # z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            z[i] = term + N.h * torch.mm(N.layers[i].weight.t(), torch.tanh(N.layers[i].forward(u[i - 1])).t() * term)

        # z_0 = K_0' diag(...) z_1
        z[0] = torch.mm(N.layers[0].weight.t(), tanhopen.t() * z[1])
        grad = z[0] + torch.mm(symA, x.t()) + self.c.weight.t()

        if just_grad:
            return grad.t()

        # -----------------
        # trace of Hessian
        # -----------------

        # t_0, the trace of the opening layer
        Kopen = N.layers[0].weight[:, 0:d]  # indexed version of Kopen = torch.mm( N.layers[0].weight, E  )
        temp = deriv_tanh(opening.t()) * z[1]
        trH = torch.sum(temp.reshape(m, -1, nex) * torch.pow(Kopen.unsqueeze(2), 2), dim=(0, 1))  # trH = t_0

        # grad_s u_0 ^ T
        temp = tanhopen.t()  # act'( K_0 * S + b_0 )
        Jac = Kopen.unsqueeze(2) * temp.unsqueeze(1)  # K_0' * act'( K_0 * S + b_0 )
        # Jac is shape m by d by nex

        # t_i, trace of the resNet layers
        # KJ is the K_i^T * grad_s u_{i-1}^T
        for i in range(1, N.n_layers):
            KJ = torch.mm(N.layers[i].weight, Jac.reshape(m, -1))
            KJ = KJ.reshape(m, -1, nex)
            if i == N.n_layers - 1:
                term = self.w.weight.t()
            else:
                term = z[i + 1]

            temp = N.layers[i].forward(u[i - 1]).t()  # (K_i * u_{i-1} + b_i)
            t_i = torch.sum((deriv_tanh(temp) * term).reshape(m, -1, nex) * torch.pow(KJ, 2), dim=(0, 1))
            trH = trH + N.h * t_i  # add t_i to the accumulate trace
            Jac = Jac + N.h * torch.tanh(temp).reshape(m, -1, nex) * KJ  # update Jacobian

        return grad.t(), trH + torch.trace(symA[0:d, 0:d])
        # indexed version of: return grad.t() ,  trH + torch.trace( torch.mm( E.t() , torch.mm(  symA , E) ) )
