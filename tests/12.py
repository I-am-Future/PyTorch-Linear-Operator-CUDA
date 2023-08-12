'''
Final tests for linearlayer forward calculation
'''

import torch
import mylinearops

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

## CHECK for Linear Layer with bias ##

A = torch.randn(40, 30, dtype=torch.float64).cuda().requires_grad_() * 100
linear = mylinearops.LinearLayer(30, 50).cuda().double()

res_my = linear(A)
res_torch = torch.matmul(A, linear.weight) + linear.bias
# print(res_torch.shape)
# print(linear.bias.shape)

print(torch.allclose(res_my, res_torch))
print(torch.max(torch.abs(res_my - res_torch)))

