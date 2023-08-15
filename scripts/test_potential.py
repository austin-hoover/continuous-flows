import time
import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cnflows as cnf
import torch


d = 2
m = 5

net = cnf.potential.OTPotential(n_layers=2, m=m, d=d)
net.N.layers[0].weight.data = 0.1 + 0.0 * net.N.layers[0].weight.data
net.N.layers[0].bias.data = 0.2 + 0.0 * net.N.layers[0].bias.data
net.N.layers[1].weight.data = 0.3 + 0.0 * net.N.layers[1].weight.data
net.N.layers[1].weight.data = 0.3 + 0.0 * net.N.layers[1].weight.data

# number of samples-by-(d+1)
x = torch.Tensor([[1.0, 4.0, 0.5], [2.0, 5.0, 0.6], [3.0, 6.0, 0.7], [0.0, 0.0, 0.0]])
y = net(x)
print(y)

# test timings
d = 400
m = 32
nex = 1000

net = cnf.potential.OTPotential(n_layers=5, m=m, d=d)
net.eval()
x = torch.randn(nex, d + 1)
y = net(x)

end = time.time()
g, h = net.grad_and_hessian_trace(x)
print("grad_and_hessian_trace takes ", time.time() - end)

end = time.time()
g = net.grad_and_hessian_trace(x, just_grad=True)
print("just_grad takes  ", time.time() - end)
