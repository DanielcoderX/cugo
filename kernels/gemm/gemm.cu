// Matrix multiplication kernel with shared-memory tiling
// C = A * B
// A: M x K, B: K x N, C: M x N

#define TILE_DIM 16

extern "C" __global__ void gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;

    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_DIM + threadIdx.x;
        if (row < M && aCol < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int bRow = t * TILE_DIM + threadIdx.y;
        if (bRow < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
