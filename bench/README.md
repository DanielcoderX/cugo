# cugo Benchmarks

Micro-benchmarks measuring raw Driver API dispatch overhead, kernel launch latency, and host-device transfer throughput on Windows (`nvcuda.dll`) without cgo.

## Hardware & Environment
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace, sm_89, 8GB VRAM)
- **CPU**: AMD Ryzen 7 7435HS (16 threads)
- **Driver**: NVIDIA Driver 616.64 / CUDA Driver 13.4 (raw 13040)
- **Go Version**: Go 1.26.0 `windows/amd64`

## Results

```
BenchmarkDriverCallOverhead-16     17293358        67.68 ns/op        16 B/op        2 allocs/op
BenchmarkKernelLaunchLatency-16      117566      9861.00 ns/op       248 B/op       10 allocs/op
BenchmarkMemcpyHtoD_1MB-16             8835     118175.00 ns/op     8873.05 MB/s
BenchmarkMemcpyHtoD_16MB-16             788    1492516.00 ns/op    11240.89 MB/s
BenchmarkMemcpyDtoH_1MB-16             6186     180604.00 ns/op     5805.93 MB/s
BenchmarkMemcpyDtoH_16MB-16             676    1775373.00 ns/op     9449.96 MB/s
```

## Analysis & CGO Comparison
1. **Dynamic Call Overhead**:
   Calling raw driver API functions via `Proc.Call` takes **~67 ns**. For comparison, a standard cgo call typically costs ~50-60 ns on modern Go versions. The difference is negligible for GPU workloads where kernel execution and memory transfers dominate by orders of magnitude.
2. **Kernel Launch Latency**:
   End-to-end `cuLaunchKernel` overhead (including typed parameter packing and validation) is **~9.8 µs**.
3. **Memory Throughput**:
   Memory transfers achieve PCIe Gen 4 bus saturation:
   - **HtoD**: **11.24 GB/s** (16MB buffers)
   - **DtoH**: **9.45 GB/s** (16MB buffers)
