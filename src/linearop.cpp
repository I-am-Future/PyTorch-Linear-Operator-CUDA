#include "utils.h"
#include "addmul_kernel.h"


torch::Tensor linear_forward(
    const torch::Tensor &input, 
    const torch::Tensor &weight, 
    const torch::Tensor &bias) 
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    TORCH_CHECK(input.size(1) == weight.size(0), "linear_forward: shape mismatch");

    // Y = X * W + b
    auto output = matmul_cuda(input, weight);

    // output = output + bias.expand_as(output); // TODO: convert to our own add kernel
    output = add_inplace_nxp_p_cuda(output, bias);

    return output;
}


/* Backward for input gradient*/
torch::Tensor linear_dinput_backward(
    const torch::Tensor &grad_output, 
    const torch::Tensor &input, 
    const torch::Tensor &weight, 
    const torch::Tensor &bias) 
{
    CHECK_INPUT(grad_output);
    CHECK_INPUT(weight);

    // dL/dX = dL/dY * W^T
    auto grad_input = matmul_cuda(grad_output, transpose_cuda(weight));

    return grad_input;
}

/* Backward for weight gradient */
torch::Tensor linear_dweight_backward(
    const torch::Tensor &grad_output, 
    const torch::Tensor &input, 
    const torch::Tensor &weight, 
    const torch::Tensor &bias) 
{
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);

    // dL/dW = X^T * dL/dY
    auto grad_weight = matmul_cuda(transpose_cuda(input), grad_output);

    return grad_weight;
}

/* Backward for bias gradient */
torch::Tensor linear_dbias_backward(
    const torch::Tensor &grad_output, 
    const torch::Tensor &input, 
    const torch::Tensor &weight, 
    const torch::Tensor &bias) 
{
    CHECK_INPUT(grad_output);

    // dL/db = sum(dL/dY, axis=0)
    // auto grad_bias = grad_output.sum(0);
    auto grad_bias = sum_axis_cuda(grad_output, 0);

    return grad_bias;
}


/* Pure matrix multiplication forward */
torch::Tensor matmul_forward(
    const torch::Tensor &A, 
    const torch::Tensor &B) 
{
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    TORCH_CHECK(A.size(1) == B.size(0), "matmul_fast_forward: shape mismatch");

    return matmul_cuda(A, B);
}


/* Backward for A gradient */
torch::Tensor matmul_dA_backward(
    const torch::Tensor &grad_output, 
    const torch::Tensor &A, 
    const torch::Tensor &B) 
{
    CHECK_INPUT(grad_output);
    // CHECK_INPUT(A);
    CHECK_INPUT(B);
    
    // dL/dB = dL/dY * B^T
    auto grad_A = matmul_cuda(grad_output, transpose_cuda(B));

    return grad_A;
}

/* Backward for B gradient */
torch::Tensor matmul_dB_backward(
    const torch::Tensor &grad_output, 
    const torch::Tensor &A, 
    const torch::Tensor &B) 
{
    CHECK_INPUT(grad_output);
    CHECK_INPUT(A);
    // CHECK_INPUT(B);
    
    // dL/dB = A^T * dL/dY
    auto grad_B = matmul_cuda(transpose_cuda(A), grad_output);

    return grad_B;
}


// wrapper for sum axis cuda
// This function is an experimental version, and it has not been optimized yet.
// This function is currently slower than the official implementation.
torch::Tensor sum_axis(
    const torch::Tensor &input, 
    const int axis) 
{
    CHECK_INPUT(input);

    return sum_axis_cuda(input, axis);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "Linear forward");
    m.def("linear_dinput_backward", &linear_dinput_backward, "Linear dinput backward");
    m.def("linear_dweight_backward", &linear_dweight_backward, "Linear dweight backward");
    m.def("linear_dbias_backward", &linear_dbias_backward, "Linear dbias backward");
    m.def("matmul_forward", &matmul_forward, "Matmul forward");
    m.def("matmul_dA_backward", &matmul_dA_backward, "Matmul dA backward");
    m.def("matmul_dB_backward", &matmul_dB_backward, "Matmul dB backward");
    m.def("sum_axis", &sum_axis, "Sum axis");
}
