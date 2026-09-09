# cugo: pure-Go CUDA Driver API bindings (no cgo)

[![Go Reference](https://pkg.go.dev/badge/github.com/DanielcoderX/cugo.svg)](https://pkg.go.dev/github.com/DanielcoderX/cugo)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v1.0.0-green.svg)](https://github.com/DanielcoderX/cugo/releases/tag/v1.0.0)

`cugo` exposes the NVIDIA CUDA Driver API directly to Go **without cgo** by dynamically loading `nvcuda.dll` at runtime via `golang.org/x/sys/windows`. 

While existing Go CUDA bindings require a C compiler, CUDA headers, and the CUDA Toolkit installed at build time, `cugo` requires **no C compiler to build or ship a Go program that uses the GPU** — only the standard NVIDIA display driver needs to be present on the target host at runtime. Cross-compiling GPU-accelerated Go binaries is completely frictionless (`CGO_ENABLED=0`).

---

## Key Features

- 🚀 **Zero Build Dependencies**: No MSVC, GCC, Clang, or CUDA Toolkit required for downstream users or consumers.
- 📦 **PTX & CUBIN Embedding**: Embed GPU kernels directly into Go binaries using Go 1.16+ `//go:embed`.
- ⚡ **Kernel Launch Reflection**: Pass arbitrary Go structs by-value or by-pointer, plus primitives (`bool`, `int8`-`int64`, `float32`/`float64`, `uintptr`) with automatic ABI marshaling.
- 📌 **Pinned Host Memory**: Page-locked allocations (`AllocHost`) with zero-copy direct GPU access and high-speed PCIe DMA transfers.
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

Verified and measured directly on physical hardware (NVIDIA GeForce RTX 4060 Laptop GPU, Go 1.26 windows/amd64). See [bench/README.md](bench/README.md) for full unedited benchmark measurements.

---


## Installation

```bash
go get github.com/DanielcoderX/cugo
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

	"github.com/DanielcoderX/cugo/driver"
	"github.com/DanielcoderX/cugo/kernels/vecadd"
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

# 16x16 shared-memory tiled GEMM
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
