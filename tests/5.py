import torch
import mylinearops_cuda
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print('Init complete!')

class Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return mylinearops_cuda.matmul_forward(A, B)

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        return mylinearops_cuda.matmul_dA_backward(grad_output.contiguous(), A, B), \
            mylinearops_cuda.matmul_dB_backward(grad_output.contiguous(), A, B)
    
A = torch.randn(10, 20).to('cuda:0').requires_grad_()
B = torch.randn(20, 30).to('cuda:0').requires_grad_()
A1 = A.clone().detach().requires_grad_()
B1 = B.clone().detach().requires_grad_()


res = Matmul.apply(A, B)
res1 = torch.mm(A1, B1)

res.sum().backward()
res1.sum().backward()

print(torch.max(torch.abs(A.grad - A1.grad)))
print(torch.max(torch.abs(B.grad - B1.grad)))
