# cugo: pure-Go CUDA Driver API bindings (no cgo)

[![Go Reference](https://pkg.go.dev/badge/github.com/cugo/cugo.svg)](https://pkg.go.dev/github.com/cugo/cugo)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v1.0.0-green.svg)](https://github.com/cugo/cugo/releases/tag/v1.0.0)

`cugo` exposes the NVIDIA CUDA Driver API directly to Go **without cgo** by dynamically loading `nvcuda.dll` at runtime via `golang.org/x/sys/windows`. 

While existing Go CUDA bindings require a C compiler, CUDA headers, and the CUDA Toolkit installed at build time, `cugo` requires **no C compiler to build or ship a Go program that uses the GPU** — only the standard NVIDIA display driver needs to be present on the target host at runtime. Cross-compiling GPU-accelerated Go binaries is completely frictionless (`CGO_ENABLED=0`).

---

## Key Features

- 🚀 **Zero Build Dependencies**: No MSVC, GCC, Clang, or CUDA Toolkit required for downstream users or consumers.
- 📦 **PTX & CUBIN Embedding**: Embed GPU kernels directly into Go binaries using Go 1.16+ `//go:embed`.
- ⚡ **Kernel Launch Reflection**: Pass arbitrary Go structs by-value or by-pointer, plus primitives (`bool`, `int8`-`int64`, `float32`/`float64`, `uintptr`) with automatic ABI marshaling.
- 📌 **Pinned Host Memory**: Page-locked allocations (`AllocHost`) with zero-copy direct GPU access and full 13 GB/s PCIe 4.0 DMA saturation.
- 🧠 **Unified Memory**: Coherent CPU/GPU virtual memory (`AllocManaged`) with automatic migration and explicit prefetching (`PrefetchToDevice`, `PrefetchToCPU`).
- 🔄 **CUDA Graphs**: Capture entire execution DAGs (`BeginCapture`, `EndCapture`, `Instantiate`, `Launch`) to execute complex pipelines with sub-microsecond launch latency.
- 🏊 **Stream-Ordered Allocator**: Modern CUDA 11.2+ memory pools (`AllocAsync`, `FreeAsync`, `TrimTo`) with zero-synchronization GPU memory recycling.
- 🛠️ **Dynamic JIT Linker**: In-process runtime compilation and linking of PTX strings directly into native device CUBIN binaries (`CreateLinker`, `AddPTX`, `Complete`).
- 📐 **2D Pitched Memory & Arrays**: Hardware-aligned stride allocations (`AllocPitch`), 2D rectangular transfers (`Copy2D`), and CUDA hardware arrays (`CreateArray2D`).
- 🌐 **Multi-GPU P2P**: Direct NVLink / PCIe peer-to-peer copies (`CanAccessPeer`, `EnablePeerAccess`, `CopyPeer`).
- 🧮 **Occupancy Auto-Tuning**: Built-in occupancy calculators (`MaxActiveBlocksPerMultiprocessor`, `SuggestBlockSize`).

---

## Feature Comparison

| Capability | `cugo` (v1.0.0) | `gorgonia/cu` |
|---|---|---|
| **CGO Required** | **No** (`CGO_ENABLED=0` friendly) | Yes |
| **Build-Time C Toolchain** | **None** | MSVC / GCC / Clang required |
| **CUDA Toolkit at Build Time** | **None** (library consumers) | Required (`cuda.h`, import libs) |
| **Cross-Compilation** | Seamless from any OS/architecture | Requires cross-compilation toolchain |
| **Driver Dependency** | Runtime `nvcuda.dll` | Runtime + link-time driver libraries |
| **Kernel Param Reflection** | **Yes** (Go structs & primitives) | Manual packing |
| **CUDA Graphs API** | **Yes** (Capture & Replay) | No |
| **Stream-Ordered MemPool** | **Yes** (`cuMemAllocAsync`) | No |
| **Runtime JIT Linker** | **Yes** (`cuLinkCreate`) | No |
| **Platform Support** | Windows (`amd64`, v1.0) | Windows, Linux |

---

## Performance & Micro-benchmarks

Benchmarked on **NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace, sm_89, 8GB VRAM)** + **AMD Ryzen 7 7435HS**:

| Metric | Measured Value | Notes |
|---|---|---|
| **Raw Driver Dispatch Overhead** | **67.68 ns/op** | Competitive with cgo (~50-60 ns) |
| **Kernel Launch Latency** | **9.86 µs/op** | End-to-end dispatch + param validation |
| **Pageable Host-to-Device (16MB)** | **11,240 MB/s** | Standard pageable transfer |
| **Pageable Device-to-Host (16MB)** | **9,450 MB/s** | Standard pageable transfer |
| **Pinned DMA Host-to-Device (16MB)**| **13,000 MB/s (13.0 GB/s)** | Full PCIe 4.0 link saturation |
| **Pinned DMA Device-to-Host (16MB)**| **12,772 MB/s (12.8 GB/s)** | Full PCIe 4.0 link saturation |
| **Shared-Memory Tiled GEMM** | **830.44 GFLOPS** | 1024x1024 matrix multiplication |

See [bench/README.md](bench/README.md) for full benchmarks and methodology.

---

## Installation

```bash
go get github.com/cugo/cugo
```

Requires Go 1.22+ and NVIDIA display drivers installed. No CUDA Toolkit or C compiler required.

---

## Quickstart Examples

### 1. Vector Addition Kernel

```go
package main

import (
	"fmt"
	"log"
	"unsafe"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/kernels/vecadd"
)

func main() {
	if err := driver.Init(); err != nil {
		log.Fatal(err)
	}
	devices, _ := driver.Devices()
	ctx, err := devices[0].CreateContext()
	if err != nil {
		log.Fatal(err)
	}
	defer ctx.Destroy()

	mod, err := ctx.LoadModuleData(vecadd.PTX)
	if err != nil {
		log.Fatal(err)
	}
	defer mod.Unload()

	fn, err := mod.Function("vecAdd")
	if err != nil {
		log.Fatal(err)
	}

	const n = 100000
	const byteSize = n * 4
	hA, hB, hC := make([]float32, n), make([]float32, n), make([]float32, n)
	for i := range hA {
		hA[i] = float32(i)
		hB[i] = float32(i) * 2
	}

	dA, _ := ctx.Alloc(byteSize); defer ctx.Free(dA)
	dB, _ := ctx.Alloc(byteSize); defer ctx.Free(dB)
	dC, _ := ctx.Alloc(byteSize); defer ctx.Free(dC)

	_ = ctx.CopyHtoD(dA, unsafe.Slice((*byte)(unsafe.Pointer(&hA[0])), byteSize))
	_ = ctx.CopyHtoD(dB, unsafe.Slice((*byte)(unsafe.Pointer(&hB[0])), byteSize))

	cfg := driver.LaunchConfig{
		GridDimX:  (n + 255) / 256,
		BlockDimX: 256,
	}
	_ = fn.Launch(cfg, dA, dB, dC, int32(n))

	_ = ctx.CopyDtoH(unsafe.Slice((*byte)(unsafe.Pointer(&hC[0])), byteSize), dC)
	fmt.Printf("GPU Result[42] = %.1f (expected %.1f)\n", hC[42], hA[42]+hB[42])
}
```

### 2. Unified Memory (Zero-Memcpy)

```go
// Allocate shared virtual memory directly accessible by CPU and GPU
mem, _ := ctx.AllocManaged(size)
defer mem.Free()

// Access directly on host CPU
slice := mem.Bytes()
slice[0] = 42

// Launch kernel directly using mem — hardware migrates pages on demand
_ = fn.Launch(cfg, mem, int32(n))
```

### 3. CUDA Graphs (Capture & Replay)

```go
stream, _ := ctx.CreateStream()
defer stream.Destroy()

// 1. Capture stream operations into a graph
_ = stream.BeginCapture()
_ = fn.Launch(cfg, dA, dB, dC, int32(n))
graph, _ := stream.EndCapture()
defer graph.Destroy()

// 2. Instantiate and launch repeatedly with sub-microsecond CPU overhead
exec, _ := graph.Instantiate()
defer exec.Destroy()

_ = exec.Launch(stream)
_ = stream.Synchronize()
```

### 4. Dynamic JIT Linker

```go
linker, _ := ctx.CreateLinker()
defer linker.Destroy()

// Feed PTX code directly generated at runtime
_ = linker.AddPTX(ptxBytes, "my_kernel.ptx")

// Compile and link directly to hardware CUBIN
cubin, _ := linker.Complete()

// Load and execute immediately
mod, _ := ctx.LoadModuleData(cubin)
defer mod.Unload()
```

---

## Included Examples & Benchmarks

```bash
# Enumerate GPU models, compute capabilities, and hardware attributes
go run ./examples/device-info

# Basic vector addition end-to-end kernel launch
go run ./examples/vecadd

# Async streams, event recording, and transfer overlap
go run ./examples/async-copy

# High-performance 16x16 shared-memory tiled GEMM (>830 GFLOPS)
go run ./examples/gemm
```

---

## Architecture & Design Documents

- [docs/design.md](docs/design.md): System architecture, fastcall calling conventions, error mapping.
- [docs/decisions.md](docs/decisions.md): Architecture Decision Records (ADR-0001 through ADR-0016).
- [docs/driver-api-coverage.md](docs/driver-api-coverage.md): Complete table of 50+ bound CUDA Driver API functions.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
