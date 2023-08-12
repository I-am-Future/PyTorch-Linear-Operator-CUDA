'''
Final tests for linearlayer backward calculation
'''

import torch
import mylinearops

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

## CHECK for Linear Layer with bias ##

A = torch.randn(40, 30, dtype=torch.float64).cuda().requires_grad_()
linear = mylinearops.LinearLayer(30, 40).cuda().double()

print(torch.autograd.gradcheck(linear, (A,)))    # pass



## CHECK for Linear Layer without bias ##

A = torch.randn(40, 30, dtype=torch.float64).cuda().requires_grad_()
linear_nobias = mylinearops.LinearLayer(30, 40, bias=False).cuda().double()

print(torch.autograd.gradcheck(linear_nobias, (A,)))    # pass
