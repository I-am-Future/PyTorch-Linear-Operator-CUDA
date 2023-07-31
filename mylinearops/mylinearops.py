import torch
import torch.nn as nn
import mylinearops_cuda
import math


class Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return mylinearops_cuda.matmul_forward(A, B)

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = None
        if ctx.needs_input_grad[0]:
            grad_A = mylinearops_cuda.matmul_dA_backward(grad_output.contiguous(), A, B)
        grad_B = None
        if ctx.needs_input_grad[1]:
            grad_B = mylinearops_cuda.matmul_dB_backward(grad_output.contiguous(), A, B)
        return grad_A, grad_B


class LinearOp(torch.autograd.Function):
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


class LinearLayer(nn.Module):
    ''' Self-implemented linear layer on CUDA, mapping `in_features` to `out_features`.
        Bias is optional. If bias is True, the module will call LinearOp, otherwise it will call Matmul.
        
        Example:
        ```
            >>> import mylinearops
            >>> linear = mylinearops.LinearLayer(10, 20, bias=True).cuda()
            >>> input = torch.randn(128, 10).cuda()
            >>> output = linear(input)
            >>> print(output.size())
            torch.Size([128, 20])
        ```
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X):
        if self.bias is None:
            return Matmul.apply(X, self.weight)
        return LinearOp.apply(X, self.weight, self.bias)


def matmul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    ''' Self-implemented matrix multiplication on CUDA. Support forward and backward. 

        Example:
        ```
            >>> import mylinearops
            >>> A = torch.randn(30, 40).cuda()
            >>> B = torch.randn(40, 50).cuda()
            >>> C = mylinearops.matmul(A, B)
            >>> print(C.size())
            torch.Size([30, 50])
        ```
    '''
    return Matmul.apply(lhs, rhs)

def linearop(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ''' Self-implemented integrated linear operation Y = XW + b on CUDA. 
        Support forward and backward. 

        Example:
        ```
            >>> import mylinearops
            >>> X = torch.randn(128, 30).cuda()
            >>> W = torch.randn(30, 40).cuda()
            >>> b = torch.randn(40).cuda()
            >>> Y = mylinearops.linearop(X, W, b)
            >>> print(Y.size())
            torch.Size([128, 40])
        ```
    '''
    return LinearOp.apply(X, W, b)

