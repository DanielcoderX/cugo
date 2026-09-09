// rmsnorm.cu - High-performance warp-shuffle fused RMSNorm kernel for cugo
#include <cuda_runtime.h>
#include <math.h>

#define WARP_SIZE 32

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C" __global__ void rmsnorm_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    int rows,
    int cols,
    float eps
) {
    // Each block processes one row
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    __shared__ float s_sum[32]; // Up to 32 warps (1024 threads)
    __shared__ float row_rms_inv;

    // Step 1: Accumulate sum of squares
    float sum_sq = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float val = row_in[col];
        sum_sq += val * val;
    }

    sum_sq = warpReduceSum(sum_sq);
    if (lane == 0) {
        s_sum[wid] = sum_sq;
    }
    __syncthreads();

    if (wid == 0) {
        float b_sum = (lane < num_warps) ? s_sum[lane] : 0.0f;
        b_sum = warpReduceSum(b_sum);
        if (lane == 0) {
            float mean_sq = b_sum / (float)cols;
            row_rms_inv = rsqrtf(mean_sq + eps);
        }
    }
    __syncthreads();

    float r_inv = row_rms_inv;

    // Step 2: Normalize and scale by weight (if weight != NULL)
    if (weight != NULL) {
        for (int col = tid; col < cols; col += blockDim.x) {
            row_out[col] = row_in[col] * r_inv * weight[col];
        }
    } else {
        for (int col = tid; col < cols; col += blockDim.x) {
            row_out[col] = row_in[col] * r_inv;
        }
    }
}
