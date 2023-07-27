
import torch

A = torch.randn(10, 20).requires_grad_()
B = torch.randn(20, 30).requires_grad_()

res = torch.mm(A, B)
res.retain_grad()
res.sum().backward()

print(res.grad)

print(torch.allclose(A.grad, torch.mm(res.grad, B.t())))
print(torch.allclose(B.grad, torch.mm(A.t(), res.grad)))
