# This file is used to test the correctness of the sum_axis

import torch
import mylinearops_cuda
import os
import time


os.environ['CUDA_VISIBLE_DEVICES'] = '5'
print('Init complete!')

# test0

a = torch.FloatTensor([[1, 2, 3], [4, 5, 6]]).to('cuda:0')
print(a)
print(mylinearops_cuda.sum_axis(a, 0))
print(mylinearops_cuda.sum_axis(a, 1))

# test1
a = torch.randn(300, 400).to('cuda:0')

t0 = time.time()
sum = mylinearops_cuda.sum_axis(a, 0)
torch.cuda.synchronize()
t1 = time.time()
sum1 = torch.sum(a, 0)
torch.cuda.synchronize()
t2 = time.time()

sum2 = mylinearops_cuda.sum_axis(a, 1)
sum3 = torch.sum(a, 1)

print(torch.max(torch.abs(sum - sum1)))
print(torch.max(torch.abs(sum2 - sum3)))
print('mylinearops_cuda.sum_axis(a, 0) time: ', t1 - t0)
print('torch.sum(a, 0) time: ', t2 - t1)

# test2: larger one
a = torch.randn(16384, 16384).to('cuda:0')

t0 = time.time()
sum = mylinearops_cuda.sum_axis(a, 0)
torch.cuda.synchronize()

t1 = time.time()
sum1 = torch.sum(a, 0)
torch.cuda.synchronize()
t2 = time.time()

sum2 = mylinearops_cuda.sum_axis(a, 1)
sum3 = torch.sum(a, 1)


print(torch.max(torch.abs(sum - sum1)))
print(torch.max(torch.abs(sum2 - sum3)))
print('mylinearops_cuda.sum_axis(a, 0) time: ', t1 - t0)
print('torch.sum(a, 0) time: ', t2 - t1)
