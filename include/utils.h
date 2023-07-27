#pragma once

#include <torch/extension.h>
#include "cuda_runtime.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


#define RUN_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error! at:\n"); \
        fprintf(stderr, "  %s\n", __FILE__); \
        fprintf(stderr, "  %d\n", __LINE__); \
        fprintf(stderr, "  %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define DIV_CEIL(a, b) (((a) + (b) - 1) / (b))

