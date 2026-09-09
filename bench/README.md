# cugo Benchmarks

Real, unedited benchmark measurements executed on physical hardware.

## Environment & Hardware

- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM, Compute Capability sm_89)
- **NVIDIA Driver**: 616.64 (CUDA UMD Version 13.4, raw driver version 13040)
- **CPU**: AMD Ryzen 7 7435HS (16 threads)
- **OS**: Windows (WDDM driver model, `windows/amd64`)
- **Go Version**: `go version go1.26.0 windows/amd64`

## Benchmark Command

```powershell
cd bench
go test -bench Benchmark .
```

## Raw Output

```
goos: windows
goarch: amd64
pkg: github.com/cugo/cugo/bench
cpu: AMD Ryzen 7 7435HS                             
BenchmarkDriverCallOverhead-16      	20473622	        57.10 ns/op	       8 B/op	       1 allocs/op
BenchmarkVecAddRoundTrip_100K-16    	    4711	    242590 ns/op	4946.62 MB/s	     352 B/op	      17 allocs/op
PASS
ok  	github.com/cugo/cugo/bench	3.150s
```

## Observations

1. **Driver Call Overhead (`BenchmarkDriverCallOverhead`)**:
   - Calling raw CUDA Driver API functions via lazy Windows dynamic procedures (`nvcuda.dll`) takes **57.10 ns/op** on Go 1.26 windows/amd64.
2. **End-to-End Vector Addition Round-Trip (`BenchmarkVecAddRoundTrip_100K`)**:
   - Complete lifecycle for 100,000 `float32` elements (400 KB input A + 400 KB input B HtoD copy, `vecAdd` kernel launch, 400 KB output DtoH copy): **242.59 µs/op** with pageable memory (~4.95 GB/s effective transfer + compute).
