# This file is used to test the correctness of mylinearops_cuda.matmul_forward 

import torch
import mylinearops_cuda
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

print('init!')

m = 4098
n = 4098
k = 4098

for _ in range(10):
    A = torch.randn(m, n).to('cuda:0')
    B = torch.randn(n, k).to('cuda:0')
    
    A1 = A.clone().cpu()
    B1 = B.clone().cpu()

    t0 = time.time()
    my_res = mylinearops_cuda.matmul_forward(A, B)
    torch.cuda.synchronize()
    t1 = time.time()
    torch_res = torch.mm(A, B)
    torch.cuda.synchronize()
    t2 = time.time()
    torch_res1 = torch.mm(A1, B1)
    t3 = time.time()

    print('our matmul:', t1 - t0,'torch cuda matmul:', t2 - t1,'torch cpu matmul:', t3 - t2, end='; ')

    print('max. error our & torch cuda', torch.max(torch.abs(my_res - torch_res)), end='; ')
    print('max. error torch cuda & torch cpu',torch.max(torch.abs(torch_res1 - torch_res.cpu())))
