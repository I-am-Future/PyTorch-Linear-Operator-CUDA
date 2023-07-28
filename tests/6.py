# This file is to test the correctness of the forward function of linear

import torch
import mylinearops_cuda
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
print('Init complete!')


X = torch.randn(10, 20).to('cuda:0').requires_grad_()
W = torch.randn(20, 30).to('cuda:0').requires_grad_()
b = torch.randn(30).to('cuda:0').requires_grad_()


y = mylinearops_cuda.linear_forward(X, W, b)

y1 = torch.mm(X, W) + b

print(torch.allclose(y, y1))
print(torch.max(torch.abs(y - y1)))


