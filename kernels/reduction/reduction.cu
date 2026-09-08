// High-throughput parallel sum reduction using warp shuffles and shared memory

#define WARP_SIZE 32

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C" __global__ void reduce_sum(
    const float* __restrict__ g_idata,
    float* __restrict__ g_odata,
    int n
) {
    __shared__ float shared[WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += g_idata[i];
    }

    sum = warpReduceSum(sum);

    if (lane == 0) {
        shared[wid] = sum;
    }
    __syncthreads();

    float finalSum = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        finalSum = warpReduceSum(finalSum);
        if (lane == 0) {
            atomicAdd(g_odata, finalSum);
        }
    }
}
