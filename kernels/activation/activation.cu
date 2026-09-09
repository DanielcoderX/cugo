extern "C" __device__ float fast_tanh(float x) {
    // 1 - 2 / (exp(2x) + 1)
    float exp2x = __expf(2.0f * x);
    return (exp2x - 1.0f) / (exp2x + 1.0f);
}

extern "C" __global__ void geluKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Fast tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        float cdf = 0.5f * (1.0f + fast_tanh(inner));
        output[idx] = x * cdf;
    }
}

extern "C" __global__ void reluKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x > 0.0f ? x : 0.0f;
    }
}

extern "C" __global__ void siluKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x / (1.0f + __expf(-x));
    }
}

extern "C" __global__ void sigmoidKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = 1.0f / (1.0f + __expf(-x));
    }
}

extern "C" __global__ void biasAddKernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        output[idx] = input[idx] + bias[col];
    }
}
