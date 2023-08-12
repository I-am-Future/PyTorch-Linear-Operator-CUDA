'''
Final tests for matmul backward calculation
'''

import torch
import mylinearops

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

A = torch.randn(20, 30, dtype=torch.float64).cuda().requires_grad_()
B = torch.randn(30, 40, dtype=torch.float64).cuda().requires_grad_()


# print(mylinearops.matmul(A, B))

# do the grad check with torch.autograd

print(torch.autograd.gradcheck(mylinearops.matmul, (A, B), eps=1e-3))    # pass
print(torch.autograd.gradcheck(mylinearops.matmul, (A, B)))    # pass



