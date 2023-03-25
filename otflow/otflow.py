import math
import torch
from torch.nn.functional import pad


def vectorize(x):
    """Vectorize tensor x."""
    return x.view(-1, 1)


def objective(x, Phi, tspan, nt, stepper="rk4", alph=[1.0, 1.0, 1.0]):
    """Evaluate objective function of OT Flow problem.
    
    See Eq. (8) in the paper.

    Parameters
    ----------
    x : tensor, shape (nex, d)
        Input data tensor.
    Phi : torch.nn.Module
        A neural network.
    tspan : list[float], shape (2,)
        Integration time range; ex: [0.0 , 1.0].
    nt : int
        The number of time steps.
    stepper : {"rk1", "rk4"}
        The Runge-Kutta scheme.
    alph : list[float], shape (3,)
        The alpha value multipliers.
    
    Returns
    -------
    Jc : float
        The objective function value dot(alph, cs).
    cs : list, shape (5,)
        The five computed costs.
    """
    h = (tspan[1]-tspan[0]) / nt

    # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
    z = pad(x, (0, 3, 0, 0), value=0)

    tk = tspan[0]

    if stepper=='rk4':
        for k in range(nt):
            z = stepRK4(odefun, z, Phi, alph, tk, tk + h)
            tk += h
    elif stepper=='rk1':
        for k in range(nt):
            z = stepRK1(odefun, z, Phi, alph, tk, tk + h)
            tk += h

    # ASSUME all examples are equally weighted
    costL  = torch.mean(z[:,-2])
    costC  = torch.mean(C(z))
    costR  = torch.mean(z[:,-1])

    cs = [costL, costC, costR]

    # return dot(cs, alph)  , cs
    return sum(i[0] * i[1] for i in zip(cs, alph)) , cs


def stepRK4(odefun, z, Phi, alph, t0, t1):
    """Runge-Kutta 4 integration scheme.
    
    Parameters
    ----------
    odefun : callable
        Function to apply at every time step.
    z : tensor, shape (nex, d+4)
        Inputs.
    Phi : torch.nn.Module
        The Phi potential function.
    alph : list[float], shape (3,)
        The 3 alpha values for the OT-Flow Problem.
    t0, t1 : float
        Start/stop time.
        
    Returns
    -------
    tensor, shape (nex, d + 4)
        The features at time t1.
    """
    h = t1 - t0 # step size
    z0 = z

    K = h * odefun(z0, t0, Phi, alph=alph)
    z = z0 + (1.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + K , t0+h , Phi, alph=alph)
    z += (1.0/6.0) * K

    return z


def stepRK1(odefun, z, Phi, alph, t0, t1):
    """Runge-Kutta 1 / Forward Euler integration scheme.  
    
    Added for comparison; stepRK4 is recommended.
    
    Parameters
    ----------
    odefun : callable
        Function to apply at every time step.
    z : tensor, shape (nex, d+4)
        Inputs.
    Phi : torch.nn.Module
        The Phi potential function.
    alph : list[float], shape (3,)
        The 3 alpha values for the mean field game problem.
    t0, t1 : float
        Start/stop time.
        
    Returns
    -------
    tensor, shape (nex, d + 4)
        The features at time t1.
    """
    z += (t1 - t0) * odefun(z, t0, Phi, alph=alph)
    return z


def integrate(
    x, network, tspan, nt, stepper="rk4", alph=[1.0, 1.0, 1.0], intermediates=False
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
    stepper : {"rk1", "rk4"}
        The Runge-Kutta scheme.
    alph : list[float], shape (3,)
        The alpha value multipliers.
    intermediates : bool
        If True, save all intermediate time points along the trajectories.
    
    Returns
    -------
    z : tensor, shape (nex, d + 4)
        The features at time t1. (Returned if `intermediates=False`.)
    zFull : tensor, shape (nex, d + 3, nt + 1)
        Trajectories from time t0 to t1. (Returned if `intermediates=True`.)
    """
    h = (tspan[1]-tspan[0]) / nt

    # initialize "hidden" vector to propagate with all the additional dimensions for all the ODEs
    z = pad(x, (0, 3, 0, 0), value=tspan[0])

    tk = tspan[0]

    if intermediates: # save the intermediate values as well
        zFull = torch.zeros( *z.shape , nt+1, device=x.device, dtype=x.dtype) # make tensor of size z.shape[0], z.shape[1], nt
        zFull[:,:,0] = z

        if stepper == 'rk4':
            for k in range(nt):
                zFull[:,:,k+1] = stepRK4(odefun, zFull[:,:,k], network, alph, tk, tk+h)
                tk += h
        elif stepper == 'rk1':
            for k in range(nt):
                zFull[:,:,k+1] = stepRK1(odefun, zFull[:,:,k], network, alph, tk, tk+h)
                tk += h

        return zFull

    else:
        if stepper == 'rk4':
            for k in range(nt):
                z = stepRK4(odefun,z, network, alph,tk,tk+h)
                tk += h
        elif stepper == 'rk1':
            for k in range(nt):
                z = stepRK1(odefun,z, network, alph,tk,tk+h)
                tk += h

        return z

    # return in case of error
    return -1


def C(z):
    """Expected negative log-likelihood; see Eq.(3) in the paper."""
    d = z.shape[1]-3
    l = z[:,d] # log-det
    return -( torch.sum(  -0.5 * math.log(2*math.pi) - torch.pow(z[:,0:d],2) / 2  , 1 , keepdims=True ) + l.unsqueeze(1) )


def odefun(x, t, network, alph=[1.0, 1.0, 1.0]):
    """Neural ODE.
    
    Combines the characteristics and log-determinant (see Eq. (2)), the 
    transport costs (see Eq. (5)), and the HJB regularizer (see Eq. (7)).

    d_t  [x ; l ; v ; r] = odefun( [x ; l ; v ; r] , t )

    x - particle position
    l - log determinant
    v - accumulated transport costs (Lagrangian)
    r - accumulates violation of HJB condition along trajectory
    """
    nex, d_extra = x.shape
    d = d_extra - 3

    z = pad(x[:, :d], (0, 1, 0, 0), value=t) # concatenate with the time t

    gradPhi, trH = network.hessian_trace(z)

    dx = -(1.0/alph[0]) * gradPhi[:,0:d]
    dl = -(1.0/alph[0]) * trH.unsqueeze(1)
    dv = 0.5 * torch.sum(torch.pow(dx, 2) , 1 ,keepdims=True)
    dr = torch.abs(  -gradPhi[:,-1].unsqueeze(1) + alph[0] * dv  ) 
    
    return torch.cat( (dx,dl,dv,dr) , 1  )