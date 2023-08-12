#include "addmul_kernel.h"

#define BLOCK_SIZE 16
#define NUM_THREADS 1024
#define EXPERIMENTAL


template <typename scalar_t>
__global__ void matmul_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> B,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> result,
    const int m, const int p
)
{
#ifdef EXPERIMENTAL
    // use shared memory technique
    __shared__ scalar_t As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t Bs[BLOCK_SIZE][BLOCK_SIZE];

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    scalar_t sum = 0;
    for (int i = 0; i < A.size(1); i += BLOCK_SIZE) {
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

        for (int j = 0; j < BLOCK_SIZE; j++) {
            sum += As[threadIdx.x][j] * Bs[j][threadIdx.y];
        }
        __syncthreads();
    }
    if (row < m && col < p) {
        result[row][col] = sum;
    }
#else
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= m || col >= p) return;

    scalar_t sum = 0;
    for (int i = 0; i < A.size(1); i++) {
        sum += A[row][i] * B[i][col];
    }
    result[row][col] = sum;
#endif
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {

    const int m = A.size(0);
    const int n = A.size(1);
    const int p = B.size(1);
    
    // Create output tensor
    auto result = torch::empty({m, p}, A.options());

    const dim3 blockSize = dim3(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 gridSize = dim3(DIV_CEIL(m, BLOCK_SIZE), DIV_CEIL(p, BLOCK_SIZE));
  
    // Call the cuda kernel launcher
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_cuda", 
    ([&] {
        matmul_fw_kernel<scalar_t><<<gridSize, blockSize>>>(
            A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            B.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            result.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            m, p
        );
    }));

    return result;
}


template <typename scalar_t>
__global__ void transpose_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> res,
    const int m, const int p
)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
#ifdef EXPERIMENTAL
    // use shared memory
    __shared__ scalar_t As[BLOCK_SIZE][BLOCK_SIZE];
    if (row < m && col < p) {
        As[threadIdx.x][threadIdx.y] = A[row][col];
    }
    __syncthreads();

    const int row2 = blockIdx.y * blockDim.y + threadIdx.x;
    const int col2 = blockIdx.x * blockDim.x + threadIdx.y;
    if (row2 < p && col2 < m) {
        res[row2][col2] = As[threadIdx.y][threadIdx.x];
    }

#else
    
    if (row >= m || col >= p) return;

    res[col][row] = A[row][col];

#endif
}


torch::Tensor transpose_cuda(const torch::Tensor A) {

    const int m = A.size(0);
    const int n = A.size(1);
    
    // Create output tensor
    auto result = torch::empty({n, m}, A.options());

    const dim3 blockSize = dim3(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 gridSize = dim3(DIV_CEIL(m, BLOCK_SIZE), DIV_CEIL(n, BLOCK_SIZE));
  
    // Call the cuda kernel launcher
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_cuda", 
    ([&] {
        transpose_kernel<scalar_t><<<gridSize, blockSize>>>(
            A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            result.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            m, n
        );
    }));

    return result;
}


template <typename scalar_t>
__global__ void add_inplace_nxp_1xp_kernel(
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> B,
    const int n, const int p
)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= n || col >= p) return;

    // move B into the shared memory
    __shared__ scalar_t Bs[BLOCK_SIZE];

    /* Bug: the following commented code has wrong dimension calculation */
    // if (threadIdx.y == 0) {
        // Bs[threadIdx.x] = B[col];
    // }
    // __syncthreads();
    // A[row][col] += Bs[threadIdx.x];

    if (threadIdx.x == 0) {
        Bs[threadIdx.y] = B[col];
    }
    __syncthreads();

    A[row][col] += Bs[threadIdx.y];

}


/*
 * A: n x p matrix, B: p-dim vector (1 x p)
 * Use broadcasting. Result store in A.
 */
torch::Tensor add_inplace_nxp_p_cuda(
    const torch::Tensor A, 
    const torch::Tensor B
) {

    const int n = A.size(0);
    const int p = A.size(1);
    
    const dim3 blockSize = dim3(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 gridSize = dim3(DIV_CEIL(n, BLOCK_SIZE), DIV_CEIL(p, BLOCK_SIZE));
  
    // Call the cuda kernel launcher
    AT_DISPATCH_FLOATING_TYPES(A.type(), "add_inplace_nxp_1xp_cuda", 
    ([&] {
        add_inplace_nxp_1xp_kernel<scalar_t><<<gridSize, blockSize>>>(
            A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            B.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            n, p
        );
    }));

    return A;
}


template <typename scalar_t>
__global__ void sum_axis_kernel(
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> res,
    const int axis, const int n, const int m
)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    scalar_t sum = 0;
    __shared__ scalar_t sums[NUM_THREADS];

    // along row
    if (axis == 0) {
        for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += blockDim.x * gridDim.x) {
            sum += A[idx][col];
        }
        sums[threadIdx.x] = sum;
        __syncthreads();

        // reduce
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sums[threadIdx.x] += sums[threadIdx.x + stride];
            }
            __syncthreads();
        }

        // write result
        if (threadIdx.x == 0) {
            res[col] = sums[0];
        }
        // if (blockIdx.x * blockDim.x < n) {
        //     if (threadIdx.x == 0) {
        //         atomicAdd(&res[col], sums[0]);
        //     }
        // }
    } else {
        // along col
        for (int idx = threadIdx.y + blockIdx.y * blockDim.y; idx < m; idx += blockDim.y * gridDim.y) {
            sum += A[row][idx];
        }
        sums[threadIdx.y] = sum;
        __syncthreads();

        // reduce
        for (int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
            if (threadIdx.y < stride) {
                sums[threadIdx.y] += sums[threadIdx.y + stride];
            }
            __syncthreads();
        }
        
        // write result
        // if (blockIdx.y * blockDim.y < m) {
        //     if (threadIdx.y == 0) {
        //         atomicAdd(&res[row], sums[0]);
        //     }
        // }
        if (threadIdx.y == 0) {
            res[row] = sums[0];
        }
    }
}


torch::Tensor sum_axis_cuda(const torch::Tensor A, int axis)
{
    const int n = A.size(0);
    const int m = A.size(1);
    
    // Create output tensor
    auto result = torch::zeros({axis == 0 ? m : n}, A.options());

    // create block, grid config
    dim3 blockSize, gridSize;
    if (axis == 0) {
        blockSize = dim3(NUM_THREADS, 1);
        // gridSize = dim3(SQRT_CEIL(n/NUM_THREADS), m);
        gridSize = dim3(1, m);
    }
    else {
        blockSize = dim3(1, NUM_THREADS);
        gridSize = dim3(n, 1);
    }
  
    // Call the cuda kernel launcher
    AT_DISPATCH_FLOATING_TYPES(A.type(), "sum_axis_cuda", 
    ([&] {
        sum_axis_kernel<scalar_t><<<gridSize, blockSize>>>(
            A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            result.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            axis, n, m
        );
    }));

    return result;
}