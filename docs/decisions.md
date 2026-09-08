# Architecture & Design Decisions (ADRs)

## ADR-0001: Pure-Go Windows Driver Loading via Lazy DLL
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: Existing Go CUDA bindings (e.g. `gorgonia/cu`) require `cgo`, a C compiler, and CUDA headers at build time.
- **Decision**: Use `golang.org/x/sys/windows.NewLazySystemDLL("nvcuda.dll")` to dynamically load NVIDIA's user-mode driver API at runtime on Windows.
- **Consequences**:
  - No C/C++ compiler or CUDA toolkit required to build or cross-compile Go binaries.
  - Runtime dependency only requires NVIDIA display drivers installed on the host machine.
  - Driver or symbol missing produces actionable Go errors (`ErrDriverNotFound`, `ErrProcNotFound`), avoiding fatal panics.

## ADR-0002: CUresult Error Enum Mapping
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: The CUDA Driver API returns integer status codes (`CUresult`).
- **Decision**: Fully generate `CUresult` constants (100+ error codes) from CUDA 13 `cuda.h` directly into `internal/nvapi/errors.go` and implement Go's `error` interface. Return `nil` when `CUresult == CUDA_SUCCESS` (0).
- **Consequences**:
  - Exact error names and descriptions matching NVIDIA documentation.
  - Pure Go tests can test error handling without GPU hardware.

## ADR-0003: Versioned Function Name Resolution
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: NVIDIA driver exports versioned variants for several APIs (e.g., `cuDeviceTotalMem_v2`, `cuCtxCreate_v2`, `cuMemAlloc_v2`).
- **Decision**: Map internal procedures directly to the standard `_v2` symbols present across all modern 64-bit drivers.
- **Consequences**:
  - Prevents subtle pointer width and 64-bit address space bugs.

## ADR-0004: Go Goroutine Thread-Affinity & Context Binding
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: In the CUDA Driver API, `CUcontext` is bound per OS host thread. Go's runtime scheduler can migrate goroutines across OS threads at any yield point.
- **Decision**: Implement `Context.EnsureCurrent()` which queries the thread's current context (`cuCtxGetCurrent`) and only sets current (`cuCtxSetCurrent`) if the calling OS thread is not already bound to this context. Call `EnsureCurrent()` across all context-dependent methods (`Alloc`, `Free`, `CopyHtoD`, `CopyDtoH`, `CreateStream`, `CreateEvent`, `LoadModuleData`).
- **Consequences**:
  - Transparent context affinity across goroutine migrations without requiring callers to manually call `runtime.LockOSThread()` for ordinary operations.
  - Benchmarks and intensive loops can still pin threads for minimal overhead.

## ADR-0005: Kernel Parameter Marshaling via Typed KernelArg & void** Array
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: `cuLaunchKernel` requires a `void** kernelParams` array where each element points to the memory buffer of the argument.
- **Decision**: Define a typed `KernelArg` interface (`Ptr`, `Int32`, `Uint32`, `Int64`, `Uint64`, `Float32`, `Float64`, `Raw`) alongside an automatic type switch for primitive types. Pack argument addresses into a slice of `unsafe.Pointer` and guard with `runtime.KeepAlive`.
- **Consequences**:
  - Safe, predictable ABI argument packing without heavy runtime reflection overhead.
  - Avoids GC pointer corruption while retaining an idiomatic Go API.

## ADR-0006: Precompiled PTX Embedding
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: cugo is a driver API binding, not a CUDA C compiler. Users and consumers of downstream libraries need to build without `nvcc`.
- **Decision**: Kernels are authored in `.cu`, compiled offline or via `go:generate nvcc -ptx` to `.ptx`, and embedded via Go's `//go:embed`.
- **Consequences**:
  - Anyone running `go build` or `go run` does not need `nvcc` installed.
  - Portable PTX is JIT-compiled by the installed NVIDIA driver at load time.

## ADR-0007: Pinned Host Memory & Zero-Copy Access
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: Standard Go memory allocated on the heap is pageable, limiting DMA copy throughput and preventing zero-copy GPU access.
- **Decision**: Wrap `cuMemAllocHost_v2`, `cuMemFreeHost`, and `cuMemHostGetDevicePointer_v2` in `driver.HostMem`. Expose `HostMem.Bytes()` for idiomatic Go slice access and `HostMem.DevicePointer()` for direct zero-copy GPU kernel execution.
- **Consequences**:
  - DMA bandwidth increases to ~13 GB/s.
  - Zero-copy execution allows GPU kernels to read and write directly to mapped host memory without separate `CopyHtoD` / `CopyDtoH` memcpy calls.

## ADR-0008: Reflection-Based Struct & Primitive Argument Marshaling
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: CUDA kernels frequently accept structs passed by value (e.g. configuration blocks, hyperparameters) or pointers to structs, as well as fine-grained primitive types (`int8`, `uint8`, `int16`, `uint16`, `bool`).
- **Decision**: Extend `toKernelArg` with direct type cases for all Go primitives and a reflection fallback (`reflect.ValueOf`) for structs and pointers to structs. For by-value structs, allocate an addressable copy (`reflect.New(rv.Type())`) and pass its pointer to `cuLaunchKernel`.
- **Consequences**:
  - Full transparency for developers passing custom Go structs directly to `fn.Launch(cfg, myStruct)` matching CUDA C `__global__ void myKernel(MyStruct s)`.
  - Zero manual packing code required for complex kernel parameter lists.

## ADR-0009: Unified Memory (Managed Memory) & Asynchronous Prefetching
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: Programs handling complex data structures benefit from a single coherent virtual memory space shared across CPU and GPU, eliminating explicit memcpy calls.
- **Decision**: Wrap `cuMemAllocManaged`, `cuMemPrefetchAsync`, and `cuMemAdvise` in `driver.ManagedMem`. Expose `Bytes()` for direct Go CPU slices, `DevicePtr()` for kernel arguments, and prefetch methods for minimizing page fault stalls.
- **Consequences**:
  - Developers can allocate shared CPU-GPU memory with a single call to `ctx.AllocManaged()`.
  - Hardware MMU migrates pages on demand; `PrefetchToDevice` and `PrefetchToCPU` enable deterministic page migration ahead of time.
