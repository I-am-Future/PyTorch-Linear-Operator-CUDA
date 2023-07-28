#pragma once

#include "utils.h"

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);


torch::Tensor transpose_cuda(const torch::Tensor A);
