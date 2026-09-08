// vecadd.cu - Vector addition kernel for cugo
extern "C" __global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
