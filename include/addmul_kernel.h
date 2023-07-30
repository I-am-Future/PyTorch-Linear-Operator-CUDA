#pragma once

#include "utils.h"

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);


torch::Tensor transpose_cuda(const torch::Tensor A);


torch::Tensor add_inplace_nxp_p_cuda(
    const torch::Tensor A, 
    const torch::Tensor B
);


torch::Tensor sum_axis_cuda(const torch::Tensor A, int axis);

