import torch
import mylinearops_cuda
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

m = 4098
n = 4098
k = 4098

A = torch.randn(m, n).to('cuda:0')
B = torch.randn(n, k).to('cuda:0')

t0 = time.time()
my_res = mylinearops_cuda.matmul_forward(A, B)
torch.cuda.synchronize()
t1 = time.time()

A = torch.randn(m, n).to('cuda:0')
B = torch.randn(n, k).to('cuda:0')

t2 = time.time()
my_fast_res = mylinearops_cuda.matmul_fast_forward(A, B)
torch.cuda.synchronize()
t3 = time.time()

# torch_res = torch.mm(A, B)
# print(torch.max(torch.abs(my_res - torch_res)))
# print(torch.max(torch.abs(my_fast_res - torch_res)))

print('mylinearops_cuda.matmul_forward: ', t1 - t0)
print('mylinearops_cuda.matmul_forward_fast: ', t3 - t2)