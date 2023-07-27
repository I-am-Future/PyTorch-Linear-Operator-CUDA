#include "addmul_kernel.h"

#define IMPROVED

template <typename scalar_t>
__global__ void matmul_fw_fast_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> B,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> result,
    const int m, const int p
)
{
    // use shared memory technique
    __shared__ scalar_t As[16][16];
    __shared__ scalar_t Bs[16][16];

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;


    scalar_t sum = 0;
    for (int i = 0; i < A.size(1); i += 16) {
        if (i + threadIdx.y < A.size(1) && row < m) {
            As[threadIdx.x][threadIdx.y] = A[row][i + threadIdx.y];
        }
        else {
            As[threadIdx.x][threadIdx.y] = 0;
        }
        if (i + threadIdx.x < B.size(0) && col < p) {
            Bs[threadIdx.x][threadIdx.y] = B[i + threadIdx.x][col];
        }
        else {
            Bs[threadIdx.x][threadIdx.y] = 0;
        }
        __syncthreads();

        for (int j = 0; j < 16; j++) {
            sum += As[threadIdx.x][j] * Bs[j][threadIdx.y];
        }
        __syncthreads();
    }
    if (row < m && col < p) {
        result[row][col] = sum;
    }

}


template <typename scalar_t>
__global__ void matmul_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> B,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> result,
    const int m, const int p
)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= m || col >= p) return;

    scalar_t sum = 0;
    for (int i = 0; i < A.size(1); i++) {
        sum += A[row][i] * B[i][col];
    }
    result[row][col] = sum;
}

torch::Tensor matmul_fw_cuda(torch::Tensor A, torch::Tensor B, bool improved) {

    const int m = A.size(0);
    const int n = A.size(1);
    const int p = B.size(1);
    
    // Create output tensor
    auto result = torch::empty({m, p}, A.options());

    const dim3 blockSize = dim3(16, 16);
    const dim3 gridSize = dim3(DIV_CEIL(m, 16), DIV_CEIL(p, 16));
  
    // Call the cuda kernel launcher
    if (improved){
        AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_fw_cuda", 
        ([&] {
            matmul_fw_fast_kernel<scalar_t><<<gridSize, blockSize>>>(
                A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                B.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                result.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                m, p
            );
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_cuda", 
        ([&] {
            matmul_fw_kernel<scalar_t><<<gridSize, blockSize>>>(
                A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                B.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                result.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                m, p
            );
        }));

    }

    return result;
}

