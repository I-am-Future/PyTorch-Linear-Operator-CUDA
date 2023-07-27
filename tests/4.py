# This file is to test correctness of backward of mylinearops.matmul_backward

import torch
import mylinearops_cuda
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

A = torch.randn(10, 20).to('cuda:0').requires_grad_()
B = torch.randn(20, 30).to('cuda:0').requires_grad_()

res = torch.mm(A, B)
res.retain_grad()
res.sum().backward()

print(A.grad)
print(res.grad)

print(torch.allclose(A.grad, mylinearops_cuda.matmul_dA_backward(res.grad, A, B)))
print(torch.allclose(B.grad, mylinearops_cuda.matmul_dB_backward(res.grad, A, B)))

print(torch.max(torch.abs(A.grad - mylinearops_cuda.matmul_dA_backward(res.grad, A, B))))
print(torch.max(torch.abs(B.grad - mylinearops_cuda.matmul_dB_backward(res.grad, A, B))))