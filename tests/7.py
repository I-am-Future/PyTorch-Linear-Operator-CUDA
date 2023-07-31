# This file is used to test the correctness of the backward function of linear

import torch
import mylinearops_cuda
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
print('Init complete!')

class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b):
        ctx.save_for_backward(X, W, b)
        return mylinearops_cuda.linear_forward(X, W, b)

    @staticmethod
    def backward(ctx, grad_output):
        X, W, b = ctx.saved_tensors
        grad_X = None
        if ctx.needs_input_grad[0]:
            grad_X = mylinearops_cuda.linear_dinput_backward(grad_output.contiguous(), X, W, b)
        grad_W = None
        if ctx.needs_input_grad[1]:
            grad_W = mylinearops_cuda.linear_dweight_backward(grad_output.contiguous(), X, W, b)
        grad_b = None
        if ctx.needs_input_grad[2]:
            grad_b = mylinearops_cuda.linear_dbias_backward(grad_output.contiguous(), X, W, b)

        return grad_X, grad_W, grad_b
    
X = torch.randn(100, 200).to('cuda:0').requires_grad_()
W = torch.randn(200, 300).to('cuda:0').requires_grad_()
b = torch.randn(300).to('cuda:0').requires_grad_()
X1 = X.clone().detach().requires_grad_()
W1 = W.clone().detach().requires_grad_()
b1 = b.clone().detach().requires_grad_()


y = Linear.apply(X, W, b)
y.sum().backward()

y1 = torch.mm(X1, W1) + b1
y1.sum().backward()

print(torch.max(torch.abs(X.grad - X1.grad)))
print(torch.max(torch.abs(W.grad - W1.grad)))
print(torch.max(torch.abs(b.grad - b1.grad)))

print(b.grad)