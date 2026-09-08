// softmax.cu - High-performance warp-shuffle safe softmax kernel for cugo
#include <cuda_runtime.h>
#include <math.h>

#define WARP_SIZE 32

__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C" __global__ void softmax_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
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

    __shared__ float s_max[32]; // up to 32 warps (1024 threads)
    __shared__ float s_sum[32];
    __shared__ float row_max;
    __shared__ float row_sum_inv;

    // Step 1: Find row maximum
    float thread_max = -1e38f;
    for (int col = tid; col < cols; col += blockDim.x) {
        thread_max = fmaxf(thread_max, row_in[col]);
    }

    thread_max = warpReduceMax(thread_max);
    if (lane == 0) {
        s_max[wid] = thread_max;
    }
    __syncthreads();

    if (wid == 0) {
        float b_max = (lane < num_warps) ? s_max[lane] : -1e38f;
        b_max = warpReduceMax(b_max);
        if (lane == 0) {
            row_max = b_max;
        }
    }
    __syncthreads();

    float r_max = row_max;

    // Step 2: Compute exp(x - r_max) sum
    float thread_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        thread_sum += expf(row_in[col] - r_max);
    }

    thread_sum = warpReduceSum(thread_sum);
    if (lane == 0) {
        s_sum[wid] = thread_sum;
    }
    __syncthreads();

    if (wid == 0) {
        float b_sum = (lane < num_warps) ? s_sum[lane] : 0.0f;
        b_sum = warpReduceSum(b_sum);
        if (lane == 0) {
            row_sum_inv = 1.0f / (b_sum + 1e-12f);
        }
    }
    __syncthreads();

    float r_sum_inv = row_sum_inv;

    // Step 3: Write normalized output
    for (int col = tid; col < cols; col += blockDim.x) {
        row_out[col] = expf(row_in[col] - r_max) * r_sum_inv;
    }
}
