struct ScaleParams {
    float factor;
    int offset;
};

extern "C" __global__ void scaleKernel(const float* in, float* out, ScaleParams p, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * p.factor + (float)p.offset;
    }
}
