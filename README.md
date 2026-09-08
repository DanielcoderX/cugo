# cugo: pure-Go CUDA Driver API bindings (no cgo)

> **Status**: Experimental / Pre-1.0 (Windows-only for v1)

`cugo` exposes the NVIDIA CUDA Driver API directly to Go **without cgo** by dynamically loading `nvcuda.dll` at runtime via `golang.org/x/sys/windows`. While existing Go CUDA bindings require a C compiler, CUDA headers, and the CUDA Toolkit toolchain installed at build time, `cugo` requires **no C compiler to build or ship a Go program that uses the GPU** — only the standard NVIDIA display driver needs to be present on the target host at runtime. This makes cross-compiling GPU-accelerated Go binaries effortless.

## Feature Comparison

| Feature | `cugo` | `gorgonia/cu` |
|---|---|---|
| **CGO Required** | **No** (`CGO_ENABLED=0` friendly) | Yes |
| **Build-Time C Toolchain** | **None** | MSVC / GCC / Clang required |
| **CUDA Toolkit at Build Time** | **None** (for library consumers) | Required (`cuda.h`, import libs) |
| **Cross-Compilation** | Seamless from any OS/architecture | Difficult (requires cross-toolchain) |
| **Driver Dependency** | Runtime `nvcuda.dll` | Runtime + link-time driver libraries |
| **Platform Support** | Windows (`amd64`, v1) | Windows, Linux |
| **API Coverage** | Focused driver subset ([Coverage](docs/driver-api-coverage.md)) | Broad driver API coverage |
| **Maturity** | New / Experimental | Mature |

## Performance & Micro-benchmarks

Benchmarked on **RTX 4060 Laptop GPU (Ada Lovelace, sm_89)** + **AMD Ryzen 7 7435HS**:

| Benchmark | Latency / Bandwidth | Notes |
|---|---|---|
| **Driver Call Overhead (`cuDeviceGetCount`)** | **~67.7 ns/op** | Competitive with cgo (~50-60 ns) |
| **Kernel Launch Latency (`cuLaunchKernel`)** | **~9.86 µs/op** | End-to-end dispatch + param packing |
| **Host-to-Device Bandwidth (16MB)** | **11,240 MB/s (11.24 GB/s)** | Saturates PCIe 4.0 link |
| **Device-to-Host Bandwidth (16MB)** | **9,450 MB/s (9.45 GB/s)** | Saturates PCIe 4.0 link |

See [bench/README.md](bench/README.md) for detailed analysis.

## API Coverage

See [docs/driver-api-coverage.md](docs/driver-api-coverage.md) for the complete list of bound and planned CUDA Driver API functions.

## Installation

```bash
go get github.com/cugo/cugo
```

Requires Go 1.22+ and an NVIDIA display driver installed at runtime. No CUDA Toolkit or C compiler required for consumers.

## Quickstart: Vector Addition Kernel

```go
package main

import (
	"fmt"
	"log"
	"math"
	"unsafe"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/kernels/vecadd"
)

func main() {
	// 1. Initialize driver & device context
	if err := driver.Init(); err != nil {
		log.Fatal(err)
	}
	devices, _ := driver.Devices()
	ctx, err := devices[0].CreateContext()
	if err != nil {
		log.Fatal(err)
	}
	defer ctx.Destroy()

	// 2. Load embedded PTX module & lookup function
	mod, err := ctx.LoadModuleData(vecadd.PTX)
	if err != nil {
		log.Fatal(err)
	}
	defer mod.Unload()

	fn, err := mod.Function("vecAdd")
	if err != nil {
		log.Fatal(err)
	}

	// 3. Prepare data & alloc GPU buffers
	const n = 100000
	const byteSize = n * 4
	hA := make([]float32, n)
	hB := make([]float32, n)
	for i := range hA {
		hA[i] = float32(i)
		hB[i] = float32(i) * 2
	}

	dA, _ := ctx.Alloc(byteSize); defer ctx.Free(dA)
	dB, _ := ctx.Alloc(byteSize); defer ctx.Free(dB)
	dC, _ := ctx.Alloc(byteSize); defer ctx.Free(dC)

	_ = ctx.CopyHtoD(dA, unsafe.Slice((*byte)(unsafe.Pointer(&hA[0])), byteSize))
	_ = ctx.CopyHtoD(dB, unsafe.Slice((*byte)(unsafe.Pointer(&hB[0])), byteSize))

	// 4. Launch kernel
	cfg := driver.LaunchConfig{
		GridDimX:  (n + 255) / 256,
		BlockDimX: 256,
	}
	if err := fn.Launch(cfg, dA, dB, dC, int32(n)); err != nil {
		log.Fatal(err)
	}

	// 5. Copy result back
	hC := make([]float32, n)
	_ = ctx.CopyDtoH(unsafe.Slice((*byte)(unsafe.Pointer(&hC[0])), byteSize), dC)

	fmt.Printf("GPU Result[42] = %.1f (expected %.1f)\n", hC[42], hA[42]+hB[42])
}
```

Run the included examples:

```bash
# 1. Device enumeration & properties
go run ./examples/device-info

# 2. End-to-end vector addition kernel
go run ./examples/vecadd

# 3. Pipelined asynchronous streams & event timing
go run ./examples/async-copy
```

## Modifying & Regenerating Kernels

The vector addition kernel is located in `kernels/vecadd/vecadd.cu`. To recompile to PTX:

```bash
cd kernels/vecadd
nvcc -ptx -o vecadd.ptx vecadd.cu
```

The precompiled PTX is checked into git, so users never need `nvcc` just to build or run Go code using `cugo`.

## Architecture & Design

- [docs/design.md](docs/design.md): System architecture, calling conventions, error mapping.
- [docs/decisions.md](docs/decisions.md): Architecture Decision Records (ADRs).
- [docs/driver-api-coverage.md](docs/driver-api-coverage.md): API coverage roadmap.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
