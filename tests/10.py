import torch
import mylinearops

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

## CHECK for Linear Layer with bias ##

A = torch.randn(40, 30).cuda().requires_grad_()
linear = mylinearops.LinearLayer(30, 40).cuda()

print(torch.autograd.gradcheck(linear, (A,), eps=1e-3, atol=1e-3, rtol=1e-3))    # pass



## CHECK for Linear Layer without bias ##

A = torch.randn(40, 30).cuda().requires_grad_()
linear_nobias = mylinearops.LinearLayer(30, 40, bias=False).cuda()

print(torch.autograd.gradcheck(linear_nobias, (A,), eps=1e-3, atol=1e-3, rtol=1e-3))    # pass
