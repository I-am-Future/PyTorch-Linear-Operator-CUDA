#include "utils.h"
#include "linear_kernel.h"
#include "addmul_kernel.h"


torch::Tensor linear_forward(
    const torch::Tensor &input, 
    const torch::Tensor &weight, 
    const torch::Tensor &bias) 
{
    // TODO

    return input;
}


/* Backward for input gradient*/
torch::Tensor linear_dinput_backward(
    const torch::Tensor &grad_output, 
    const torch::Tensor &input, 
    const torch::Tensor &weight, 
    const torch::Tensor &bias) 
{
    // TODO

    return grad_output;
}

/* Backward for weight gradient */
torch::Tensor linear_dweight_backward(
    const torch::Tensor &grad_output, 
    const torch::Tensor &input, 
    const torch::Tensor &weight, 
    const torch::Tensor &bias) 
{
    // TODO

    return grad_output;
}

/* Backward for bias gradient */
torch::Tensor linear_dbias_backward(
    const torch::Tensor &grad_output, 
    const torch::Tensor &input, 
    const torch::Tensor &weight, 
    const torch::Tensor &bias) 
{
    // TODO

    return grad_output;
}


/* Pure matrix multiplication forward */
torch::Tensor matmul_forward(
    const torch::Tensor &A, 
    const torch::Tensor &B) 
{
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    TORCH_CHECK(A.size(1) == B.size(0), "matmul_fast_forward: shape mismatch");

    return matmul_fw_cuda(A, B, false);
}

torch::Tensor matmul_fast_forward(
    const torch::Tensor &A, 
    const torch::Tensor &B) 
{
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    TORCH_CHECK(A.size(1) == B.size(0), "matmul_fast_forward: shape mismatch");

    return matmul_fw_cuda(A, B, true);
}

/* Backward for A gradient */
torch::Tensor matmul_dA_backward(
    const torch::Tensor &grad_output, 
    const torch::Tensor &A, 
    const torch::Tensor &B) 
{
    // TODO

    return grad_output;
}

/* Backward for B gradient */
torch::Tensor matmul_dB_backward(
    const torch::Tensor &grad_output, 
    const torch::Tensor &A, 
    const torch::Tensor &B) 
{
    // TODO

    return grad_output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "Linear forward");
    m.def("linear_dinput_backward", &linear_dinput_backward, "Linear dinput backward");
    m.def("linear_dweight_backward", &linear_dweight_backward, "Linear dweight backward");
    m.def("linear_dbias_backward", &linear_dbias_backward, "Linear dbias backward");
    m.def("matmul_forward", &matmul_forward, "Matmul forward");
    m.def("matmul_fast_forward", &matmul_fast_forward, "Matmul fast forward");
    m.def("matmul_dA_backward", &matmul_dA_backward, "Matmul dA backward");
    m.def("matmul_dB_backward", &matmul_dB_backward, "Matmul dB backward");
}
