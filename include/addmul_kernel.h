#pragma once

#include "utils.h"

torch::Tensor matmul_fw_cuda(torch::Tensor A, torch::Tensor B);


torch::Tensor transpose_cuda(const torch::Tensor A);
