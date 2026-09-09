extern "C" __device__ float fast_tanh_dev(float x) {
    float exp2x = __expf(2.0f * x);
    return (exp2x - 1.0f) / (exp2x + 1.0f);
}

// GELU Backward: dx = grad_out * (0.5 * (1 + tanh(u)) + 0.5 * x * (1 - tanh^2(u)) * k * (1 + 3 * 0.044715 * x^2))
// u = k * (x + 0.044715 * x^3), k = 0.7978845608
extern "C" __global__ void geluBackwardKernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_in,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float go = grad_out[idx];

        const float k = 0.7978845608f;
        const float a = 0.044715f;
        float x2 = x * x;
        float x3 = x2 * x;

        float u = k * (x + a * x3);
        float th = fast_tanh_dev(u);
        float dth = 1.0f - th * th;

        float du_dx = k * (1.0f + 3.0f * a * x2);
        float dy_dx = 0.5f * (1.0f + th) + 0.5f * x * dth * du_dx;

        grad_in[idx] = go * dy_dx;
    }
}

// Bias Backward: column-wise reduction summing grad_out over rows into grad_bias
extern "C" __global__ void biasBackwardKernel(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_bias,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int r = 0; r < rows; r++) {
            sum += grad_out[r * cols + col];
        }
        grad_bias[col] = sum;
    }
}

// Matrix transpose for backward GEMM: B[col * rows + row] = A[row * cols + col]
extern "C" __global__ void transposeKernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int rows,
    int cols
) {
    __shared__ float tile[16][17]; // 17 avoids bank conflicts

    int x = blockIdx.x * 16 + threadIdx.x;
    int y = blockIdx.y * 16 + threadIdx.y;

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    }
    __syncthreads();

    x = blockIdx.y * 16 + threadIdx.x;
    y = blockIdx.x * 16 + threadIdx.y;

    if (x < rows && y < cols) {
        out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// RMSNorm Backward
// dx = (w * dy) / rms - (x / (N * rms^3)) * sum_j (w_j * dy_j * x_j)
// dw = sum_rows (dy * (x / rms))
extern "C" __global__ void rmsnormBackwardKernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ grad_in,
    float* __restrict__ grad_weight,
    int rows,
    int cols,
    float eps
) {
    int r = blockIdx.x; // One block per row
    int tid = threadIdx.x;

    extern __shared__ float sdata[];
    float* s_sum_sq = sdata;
    float* s_dot = sdata + blockDim.x;

    // 1. Compute row mean-square: sum(x^2)
    float local_sq = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float val = input[r * cols + c];
        local_sq += val * val;
    }
    s_sum_sq[tid] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    float mean_sq = s_sum_sq[0] / (float)cols;
    float rms = rsqrtf(mean_sq + eps); // 1 / sqrt(mean_sq + eps)
    float rms3 = rms * rms * rms;

    // 2. Compute dot product: sum(dy * w * x)
    float local_dot = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float x = input[r * cols + c];
        float dy = grad_out[r * cols + c];
        float w = (weight != 0) ? weight[c] : 1.0f;
        local_dot += dy * w * x;
    }
    s_dot[tid] = local_dot;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_dot[tid] += s_dot[tid + stride];
        }
        __syncthreads();
    }
    float sum_dot = s_dot[0];
    __syncthreads();

    // 3. Compute grad_in: dx_i = (w_i * dy_i) * rms - x_i * (sum_dot / cols) * rms3
    for (int c = tid; c < cols; c += blockDim.x) {
        float x = input[r * cols + c];
        float dy = grad_out[r * cols + c];
        float w = (weight != 0) ? weight[c] : 1.0f;

        float dx = (w * dy) * rms - (x / (float)cols) * sum_dot * rms3;
        grad_in[r * cols + c] = dx;

        if (grad_weight != 0) {
            // Atomic add across rows for weight gradient
            atomicAdd(&grad_weight[c], dy * x * rms);
        }
    }
}
