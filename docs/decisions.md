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
  - Eliminates intermediate driver staging buffers during DMA transfers.
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

## ADR-0010: Multi-GPU Peer-to-Peer Access
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: Multi-GPU hosts running parallel compute tasks need high-speed direct GPU-to-GPU memory copies without staging through host RAM.
- **Decision**: Expose `Device.CanAccessPeer()`, `Context.EnablePeerAccess()`, and `driver.CopyPeer()` / `driver.CopyPeerAsync()`.
- **Consequences**:
  - Enables direct NVLink / PCIe P2P DMA transfers between distinct GPU contexts.
  - Returns clear errors on unsupported hardware links.

## ADR-0011: Kernel Occupancy Calculation & Auto-Tuning
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: Choosing optimal thread block sizes manually requires knowing GPU architecture limits, register pressure, and shared memory usage per kernel.
- **Decision**: Bind `cuOccupancyMaxActiveBlocksPerMultiprocessor` and `cuOccupancyMaxPotentialBlockSize` to `Function.MaxActiveBlocksPerMultiprocessor` and `Function.SuggestBlockSize`.
- **Consequences**:
  - Applications can dynamically compute optimal block and grid dimensions on any user GPU at runtime.

## ADR-0012: CUDA Graphs Stream Capture & Execution Replay
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: Dispatching iterative sequences of small kernels and copies accumulates host CPU launch overhead (~10 µs per launch).
- **Decision**: Wrap `cuStreamBeginCapture_v2`, `cuStreamEndCapture`, `cuGraphInstantiate_v2`, and `cuGraphLaunch` in `Stream` and `Graph` / `GraphExec` objects.
- **Consequences**:
  - Entire execution topologies can be captured once from Go and replayed with sub-microsecond latency.

## ADR-0013: Stream-Ordered Memory Allocator & Memory Pools
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: Traditional `cuMemAlloc` and `cuMemFree` synchronize the GPU context, introducing multi-microsecond stalls in stream pipelines.
- **Decision**: Expose `Context.AllocAsync` and `Context.FreeAsync` backed by CUDA stream-ordered memory pools (`cuDeviceGetDefaultMemPool`, `cuMemPoolTrimTo`).
- **Consequences**:
  - Zero synchronization on allocation and deallocation; memory is reused immediately within the same stream.

## ADR-0014: Dynamic JIT Linker API
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: Applications generating PTX code dynamically at runtime or linking modular PTX components need on-the-fly compilation to device-native CUBIN bytecode without running external processes.
- **Decision**: Expose `Context.CreateLinker()`, `Linker.AddPTX()`, `Linker.AddCubin()`, and `Linker.Complete()` wrapping `cuLinkCreate_v2`, `cuLinkAddData_v2`, `cuLinkComplete`, and `cuLinkDestroy`.
- **Consequences**:
  - Direct in-process JIT compilation and linking of PTX strings/files to native hardware cubin binaries.
  - Returned cubin bytecode is immediately loadable via `Context.LoadModuleData()`.

## ADR-0015: 2D Pitched Memory Allocations and Strided Copies
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: 2D matrices and image processing kernels require row-aligned stride allocations to guarantee coalesced memory transactions on hardware.
- **Decision**: Wrap `cuMemAllocPitch_v2`, `cuMemcpy2D_v2`, `cuMemcpy2DAsync_v2`, `cuArrayCreate_v2`, and `cuArrayDestroy` in `Context.AllocPitch`, `Context.Copy2D`, `Stream.Copy2DAsync`, and `Context.CreateArray2D`.
- **Consequences**:
  - Automatically adheres to hardware pitch alignment constraints.
  - Rectangular memory copies handle differing source and destination pitches seamlessly without manual pointer arithmetic.

## ADR-0016: Shared-Memory Tiled Matrix Multiplication (GEMM)
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: Real-world GPU compute workloads need demonstrations beyond 1D element-wise kernels to showcase memory coalescing, 2D launch configurations, and shared memory tiling.
- **Decision**: Provide `kernels/gemm` with a 16x16 shared-memory tiled matrix multiplication kernel and benchmark example (`examples/gemm`).
- **Consequences**:
  - Serves as an end-to-end blueprint for high-throughput compute kernels.

## ADR-0017: Warp-Shuffle Parallel Reduction Kernel
- **Status**: Accepted
- **Context**: Reduction (sum, max, min) is a fundamental building block for AI and numerical workloads. Naive global memory atomic adds cause extreme serialization.
- **Decision**: Implemented `kernels/reduction` leveraging fast warp shuffle intrinsics (`__shfl_down_sync`) and shared memory block reduction, followed by atomic add on partial sums.
- **Consequences**:
  - Achieves single-kernel parallel reduction using hardware register shuffling.


## ADR-0018: Bindless Texture and Surface Objects
- **Status**: Accepted
- **Context**: Modern CUDA graphics, ray tracing, and spatial interpolation workloads use hardware texture and surface units rather than legacy texture references.
- **Decision**: Added pure-Go bindings for `cuTexObjectCreate`, `cuTexObjectDestroy`, `cuSurfObjectCreate`, and `cuSurfObjectDestroy` with 64-bit alignment verified `CUDA_RESOURCE_DESC` and `CUDA_TEXTURE_DESC` structs. Added first-class kernel launch support for `*TextureObject` and `*SurfaceObject`.
- **Consequences**:
  - Go applications can take advantage of hardware bilinear filtering, boundary clamping, and spatial texture caches.

## ADR-0019: Linux ABI Port via PureGo
- **Status**: Accepted
- **Context**: `cugo` was initially Windows-only (`nvcuda.dll` via `syscall.NewLazyDLL`). To enable Linux server and container deployment (WSL2, NVIDIA DGX, Kubernetes GPU nodes) without cgo, dynamic loading of `libcuda.so.1` is needed.
- **Decision**: Unified `internal/nvapi` under `procInvoker` interface. On Windows, uses `windows.LazyProc`. On Linux, uses `purego.RegisterLibFunc` / `purego.SyscallN` to bind `libcuda.so.1` symbols with `CGO_ENABLED=0`.
- **Consequences**:
  - 100% pure-Go cross-compilation (`GOOS=linux GOARCH=amd64 CGO_ENABLED=0`).
  - Zero C toolchain or headers required for builds.

## ADR-0020: CUDA Graph Node Inspection and Dynamic Parameter Updates
- **Status**: Accepted
- **Context**: Re-capturing or re-instantiating CUDA graphs when kernel parameters or pointers change incurs high CPU latency. Real-time inference pipelines need zero-overhead parameter updates without altering graph topology.
- **Decision**: Implemented `cuGraphGetNodes`, `cuGraphNodeGetType`, and `cuGraphExecKernelNodeSetParams` exposed as `Graph.Nodes()` and `GraphExec.SetKernelNodeParams()`.
- **Consequences**:
  - Allows swapping input/output pointers and scalar arguments in-place on instantiated graphs.
  - Sub-microsecond parameter updates.

## ADR-0021: Warp-Shuffle Numerically Stable Softmax Kernel
- **Status**: Accepted
- **Context**: Transformer models and deep learning classifiers require row-wise softmax across attention and logit tensors. Standard global memory reductions cause thread divergence and multiple kernel passes.
- **Decision**: Implemented `kernels/softmax` using single-pass online softmax with warp shuffles (`__shfl_down_sync`) and shared memory block reduction.
- **Consequences**:
  - Evaluated on matrix batches (up to 128x1024), achieving exact match against CPU reference.

## ADR-0022: Multi-Stream 3-Stage Overlapped Pipeline
- **Status**: Accepted
- **Context**: Moving data between CPU and GPU over PCIe 4.0 often creates a bottleneck if host-to-device, compute, and device-to-host operations run sequentially.
- **Decision**: Implemented `examples/pipeline` using page-locked pinned host memory (`AllocHost`) and multiple concurrent streams to overlap HtoD copy, kernel compute, and DtoH copy across chunks.
- **Consequences**:
  - Overlaps host-to-device transfer, kernel execution, and device-to-host transfer across multiple streams.


## ADR-0023: Fused Warp-Shuffle RMSNorm Kernel
- **Status**: Accepted
- **Context**: Modern LLMs (e.g. LLaMA, Mistral, Gemma) replace LayerNorm with RMSNorm to eliminate mean-centering overhead. A separate kernel for sum-of-squares and normalization incurs extra global memory roundtrips.
- **Decision**: Implemented `kernels/layernorm` with a single-pass fused RMSNorm kernel using warp shuffles (`__shfl_down_sync`) for warp-level reduction and shared memory for block-level reduction. Supports optional affine weight scaling in the same pass.
- **Consequences**:
  - Verified across configurations up to 64x1024 against CPU double-precision reference.

## ADR-0024: Asynchronous Memset and GPU-Side Stream Event Synchronization
- **Status**: Accepted
- **Context**: Buffer initialization and inter-stream dependencies often cause CPU thread contention if handled via synchronous host calls (`cudaMemset`, `cudaStreamSynchronize`).
- **Decision**: Implemented `cuMemsetD8Async`, `cuMemsetD32Async`, and `cuStreamWaitEvent` with typed wrappers on `Stream` and `Context`.
- **Consequences**:
  - Enables GPU hardware engines to initialize memory and coordinate stream dependencies with zero CPU intervention.

## ADR-0025: Multi-GPU NVLink Topology Matrix & Benchmark
- **Status**: Accepted
- **Context**: Heterogeneous multi-GPU workstations and servers often have asymmetric interconnects (some GPUs connected via NVLink, others via PCIe). Users need programmatic visibility into P2P performance.
- **Decision**: Added `examples/p2p-topology` which queries bidirectional accessibility matrices, NVLink performance rank attributes, and benchmarks peer-to-peer DMA bandwidth.
- **Consequences**:
  - Provides a single-command inspection tool for multi-GPU interconnect readiness.

## ADR-0026: Warp-Level GEMV Matrix-Vector Kernel
- **Status**: Accepted
- **Context**: Autoregressive LLM inference generation (decode phase) is strictly memory-bandwidth bound ($M=1$, vector-matrix product $y = A x$). 2D block tiled GEMMs designed for compute-bound matrix-matrix multiply suffer low SM occupancy on vector operands.
- **Decision**: Implemented `kernels/gemv` using a warp-per-row dispatch with coalesced 32-bit stride loops and `__shfl_down_sync` warp reduction to maximize memory bus saturation.
- **Consequences**:
  - Handles rectangular and large LLM matrices (e.g. 2048x4096) with near-peak memory bandwidth.

## ADR-0027: Stream Priority Range & Hardware Scheduling Control
- **Status**: Accepted
- **Context**: Real-time graphics and low-latency inference pipelines require high-priority work (e.g. head tracking or immediate token generation) to preempt background DMA or async transfers on the GPU SMs.
- **Decision**: Implemented `cuCtxGetStreamPriorityRange` and `cuStreamCreateWithPriority` exposed via `Context.StreamPriorityRange()` and `Context.CreateStreamWithPriority()`.
- **Consequences**:
  - Developers can allocate prioritized streams that preempt lower-priority compute tasks at the GPU hardware scheduler level.

## ADR-0028: Multi-GPU Tensor Parallelism Column Sharding
- **Status**: Accepted
- **Context**: LLM parameter counts exceed single-GPU VRAM and require splitting linear layers across GPUs (Megatron-LM column-parallel pattern $Y = [X W_1 \mid X W_2]$).
- **Decision**: Added `examples/tensor-parallel` showing column-wise weight sharding with parallel stream/context execution and concatenation.
- **Consequences**:
  - Validates exact mathematical equivalence (0.000000e+00 diff) between sharded and unsharded GEMM executions.








