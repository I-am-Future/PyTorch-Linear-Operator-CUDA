#pragma once

#include "utils.h"

torch::Tensor matmul_fw_cuda(torch::Tensor A, torch::Tensor B, bool improved);



