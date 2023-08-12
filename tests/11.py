'''
Final tests for matmul forward calculation
'''

import torch
import mylinearops

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

print('init!')

A = torch.randn(20, 30, dtype=torch.float64).cuda().requires_grad_()
B = torch.randn(30, 40, dtype=torch.float64).cuda().requires_grad_()


res_my = mylinearops.matmul(A, B)
res_torch = torch.matmul(A, B)

print(torch.allclose(res_my, res_torch))

