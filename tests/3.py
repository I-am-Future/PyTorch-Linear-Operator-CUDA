# This file is used to test the correctness of mylinearops_cuda.matmul_forward 

import torch
import mylinearops_cuda
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

m = 4098
n = 4098
k = 4098

for _ in range(10):
    A = torch.randn(m, n).to('cuda:0')
    B = torch.randn(n, k).to('cuda:0')

    t0 = time.time()
    my_res = mylinearops_cuda.matmul_forward(A, B)
    torch.cuda.synchronize()
    t1 = time.time()

    print(t1 - t0, end=' ')

    torch_res = torch.mm(A, B)
    print(torch.max(torch.abs(my_res - torch_res)))

