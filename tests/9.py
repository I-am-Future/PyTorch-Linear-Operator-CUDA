import torch
import mylinearops

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

A = torch.randn(2, 3, dtype=torch.float64).cuda().requires_grad_()
B = torch.randn(3, 4, dtype=torch.float64).cuda().requires_grad_()


print(mylinearops.matmul(A, B))

# do the grad check with torch.autograd

print(torch.autograd.gradcheck(mylinearops.matmul, (A, B), eps=1e-3))    # pass
print(torch.autograd.gradcheck(mylinearops.matmul, (A, B)))    # pass



