extern "C" __global__ void adamwKernel(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ exp_avg_m,
    float* __restrict__ exp_avg_v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float p = params[idx];
        float g = grads[idx];
        float m = exp_avg_m[idx];
        float v = exp_avg_v[idx];

        // 1. Decoupled weight decay (AdamW)
        p -= lr * weight_decay * p;

        // 2. Update biased first and second moments
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * (g * g);

        // 3. Compute bias-corrected moments
        float m_hat = m / bias_correction1;
        float v_hat = v / bias_correction2;

        // 4. Update parameter
        p -= lr * (m_hat / (sqrtf(v_hat) + eps));

        // 5. Store back
        params[idx] = p;
        exp_avg_m[idx] = m;
        exp_avg_v[idx] = v;
    }
}

extern "C" __global__ void sgdKernel(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ velocity,
    float lr,
    float momentum,
    float weight_decay,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float p = params[idx];
        float g = grads[idx];

        // Apply weight decay
        if (weight_decay != 0.0f) {
            g += weight_decay * p;
        }

        // Apply momentum
        if (velocity != 0 && momentum != 0.0f) {
            float v = velocity[idx];
            v = momentum * v + g;
            velocity[idx] = v;
            g = v;
        }

        // Parameter update
        p -= lr * g;
        params[idx] = p;
    }
}
