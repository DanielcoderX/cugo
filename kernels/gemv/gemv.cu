// gemv.cu - High-performance warp-level GEMV kernel for cugo
#include <cuda_runtime.h>

#define WARP_SIZE 32

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C" __global__ void gemv_forward_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int n,
    float alpha,
    float beta
) {
    // 4 warps (128 threads) or 8 warps (256 threads) per block
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int num_warps_per_block = blockDim.x / WARP_SIZE;
    int row = blockIdx.x * num_warps_per_block + warp_id_in_block;

    if (row >= m) return;

    int lane = threadIdx.x % WARP_SIZE;
    const float* row_A = A + row * n;

    float sum = 0.0f;
    for (int col = lane; col < n; col += WARP_SIZE) {
        sum += row_A[col] * x[col];
    }

    sum = warpReduceSum(sum);

    if (lane == 0) {
        if (beta == 0.0f) {
            y[row] = alpha * sum;
        } else {
            y[row] = alpha * sum + beta * y[row];
        }
    }
}
