<USER_REQUEST>
# Project: cugo — pure-Go CUDA Driver API bindings (no cgo)

You are starting a new open-source Go project called **cugo**. This document is your
complete brief. Read it fully before writing any code. Where a decision isn't
specified, make a reasonable one, document it in `docs/decisions.md`, and proceed.

## 1. Mission

Expose the NVIDIA CUDA Driver API to Go **without cgo**, by dynamically loading
`nvcuda.dll` (Windows) / `libcuda.so` (Linux, later) at runtime and calling its
exported C-ABI functions directly via `golang.org/x/sys/windows` (`NewLazySystemDLL`
/ `Proc.Call`). The existing Go CUDA binding (`gorgonia/cu`) requires cgo and the CUDA
headers/toolchain at build time. cugo's differentiator: **no C compiler required to
build or ship a Go program that uses the GPU** — only the NVIDIA driver (which the
user already has if they have a GPU) needs to be present at runtime. Cross-compiling
a cugo-based binary from any host becomes possible, since there's nothing to link.

This is a **driver-API binding library**, not a deep learning framework, not cuDNN/
cuBLAS wrappers, and not a kernel-authoring DSL. It gives Go programs the ability to:
allocate/copy GPU memory, load precompiled kernels (PTX or cubin), and launch them.

## 2. Platform scope

- **v1 target: Windows only**, since that's the dev/test environment (RTX 4060,
  8GB, Ada Lovelace / compute capability 8.9). Use build tags (`//go:build windows`)
  everywhere so a Linux port later (via `libcuda.so` + `syscall.NewProc`-equivalent
  or `purego`) is additive, not a rewrite.
- Requires an NVIDIA driver with `nvcuda.dll` present. No CUDA Toolkit needed at
  **runtime** for consumers of the library — only `cugo` itself, at **development
  time**, needs `nvcc` on the PATH (see kernel compilation below).
- Detect missing/incompatible driver at `Init()` time with a clear error, not a
  crash — dynamic loading means "DLL not found" and "symbol not found" are ordinary
  runtime errors here, not build failures. Handle both.

## 3. Kernel compilation story

Kernels are written in CUDA C (`.cu` files) and compiled to PTX or cubin via `nvcc`
at **build time**, not runtime. Approach:

- A `go:generate`-driven build step (or a small `cugo-build` CLI tool shipped as
  part of the repo) that shells out to `nvcc --ptx` (portable, JIT-compiled by the
  driver at load time — safer default) or `nvcc -cubin -arch=sm_89` (precompiled for
  a specific architecture — faster load, less portable) and embeds the result via
  `//go:embed`.
- Ship at least one real example kernel (vector add, or a simple SAXPY) compiled
  this way, with the `.cu` source, the generate directive, and the embedded PTX
  checked into the repo so `go build` works without `nvcc` present for anyone just
  trying the examples — only contributors modifying kernels need the toolkit
  installed.
- Document clearly: **cugo does not compile CUDA C itself.** It loads and launches
  whatever PTX/cubin you hand it. This is a hard boundary — don't scope-creep into
  writing a CUDA compiler.

## 4. Architecture

```
cugo/
├── internal/nvapi/        # raw ABI layer: NewLazyDLL bindings for every driver
│                          # API function used, struct layouts (CUdeviceptr,
│                          # CUcontext, CUresult codes, etc.), matching
│                          # cuda.h exactly
├── driver/                # typed, idiomatic Go wrappers over internal/nvapi
│   ├── device.go          # cuInit, cuDeviceGetCount, cuDeviceGetName, etc.
│   ├── context.go         # cuCtxCreate, cuCtxDestroy, cuCtxSetCurrent
│   ├── memory.go          # cuMemAlloc, cuMemFree, cuMemcpyHtoD, cuMemcpyDtoH
│   ├── module.go          # cuModuleLoadData (from PTX/cubin bytes), cuModuleGetFunction
│   ├── launch.go          # cuLaunchKernel + kernel-parameter marshaling helpers
│   ├── stream.go          # cuStreamCreate/Destroy/Synchronize, async variants
│   └── event.go           # cuEventCreate/Record/Synchronize/ElapsedTime
├── kernels/
│   └── vecadd/            # example kernel: vecadd.cu + go:generate + embedded PTX
├── examples/
│   ├── device-info/       # enumerate GPUs, print name/compute-capability/memory
│   ├── vecadd/             # end-to-end: alloc, copy, launch, copy back, verify
│   └── async-copy/         # streams + events, overlap compute and transfer
├── bench/                  # cugo vs gorgonia/cu (cgo) — call overhead, launch
│                           # latency, H2D/D2H bandwidth
└── docs/
    ├── design.md
    ├── decisions.md
    └── driver-api-coverage.md  # table of which CUDA Driver API functions are
                                  # bound so far vs the full API surface
```

### Core types (design these first)

```go
package driver

type Device struct { handle int32 }
func Devices() ([]Device, error)
func (d Device) Name() (string, error)
func (d Device) ComputeCapability() (major, minor int, error)
func (d Device) TotalMemory() (uint64, error)

type Context struct { /* CUcontext handle */ }
func (d Device) CreateContext() (*Context, error)
func (c *Context) Destroy() error
func (c *Context) SetCurrent() error

type DevicePtr uintptr // mirrors CUdeviceptr
func (c *Context) Alloc(size uint64) (DevicePtr, error)
func (c *Context) Free(ptr DevicePtr) error
func (c *Context) CopyHtoD(dst DevicePtr, src []byte) error
func (c *Context) CopyDtoH(dst []byte, src DevicePtr) error

type Module struct { /* CUmodule handle */ }
func (c *Context) LoadModuleData(ptxOrCubin []byte) (*Module, error)
func (m *Module) Function(name string) (*Function, error)

type Function struct { /* CUfunction handle */ }
type LaunchConfig struct {
    GridDimX, GridDimY, GridDimZ    uint32
    BlockDimX, BlockDimY, BlockDimZ uint32
    SharedMemBytes                  uint32
    Stream                          *Stream // nil = default stream
}
func (f *Function) Launch(cfg LaunchConfig, params ...any) error
```

The trickiest part of this whole project is `Launch`'s parameter marshaling —
`cuLaunchKernel` takes a `void**` array of pointers to each kernel argument. Get
this right early with the vecadd example (three `DevicePtr` args + one `int` count)
before trying to generalize to arbitrary struct/type params. Consider a
`KernelArg` interface with an explicit constructor (`driver.Ptr(devPtr)`,
`driver.Int32(5)`) rather than reflection-based `any` marshaling for v1 — reflection
can come later once the explicit version works and you understand the ABI edge cases.

### Windows dynamic loading pattern

```go
package nvapi

import "golang.org/x/sys/windows"

var (
    nvcuda    = windows.NewLazySystemDLL("nvcuda.dll")
    procInit  = nvcuda.NewProc("cuInit")
    procDeviceGetCount = nvcuda.NewProc("cuDeviceGetCount")
    // ... one NewProc per driver API function actually used
)

func Init(flags uint32) error {
    r, _, _ := procInit.Call(uintptr(flags))
    return resultToError(CUresult(r))
}
```

Every wrapped function needs: correct `uintptr` argument packing (watch out for
64-bit values on a 32-bit-arg-passing convention — CUDA driver API structs like
`CUdeviceptr` are pointer-sized, fine on amd64 but be deliberate about it), and a
`CUresult -> error` mapping (`internal/nvapi/errors.go`, generate this table from
`cuda.h`'s `CUresult` enum — there are ~100 error codes, don't hand-roll a partial
list and hope).

## 5. Milestones (build in this order)

1. **Load nvcuda.dll, call `cuInit`, call `cuDeviceGetCount`.** Prove the dynamic
   loading + calling convention works at all before anything else. This is the
   `examples/device-info` skeleton.
2. **Device enumeration + properties.** Name, compute capability, total memory —
   read-only calls, no context/memory management yet, lowest-risk way to validate
   more of the ABI surface.
3. **Context + memory round-trip.** Create context, `cuMemAlloc`, `cuMemcpyHtoD`,
   `cuMemcpyDtoH`, `cuMemFree`. Verify a byte buffer survives a round trip to GPU
   memory and back unchanged.
4. **nvcc build pipeline + PTX embedding.** Get the `vecadd.cu` -> PTX ->
   `//go:embed` pipeline working, even before you can launch it.
5. **Module load + function lookup.** `cuModuleLoadData` on the embedded PTX,
   `cuModuleGetFunction("vecAdd")`. Prove you can find the kernel entry point.
6. **Kernel launch — the hard part.** Get `cuLaunchKernel` working end-to-end with
   the vecadd example: alloc three device buffers, copy inputs, launch, copy result
   back, verify against a CPU-computed expected result. This is the milestone that
   proves the whole concept works.
7. **Streams + events.** Async copy/launch, `cuStreamSynchronize`,
   `cuEventElapsedTime` for timing. Build `examples/async-copy`.
8. **Error handling pass.** Full `CUresult` enum mapped, driver-missing and
   symbol-missing cases produce clear errors, not panics or cryptic syscall failures.
9. **Benchmarks vs gorgonia/cu (cgo).** Measure call overhead (`Proc.Call` dynamic
   dispatch vs cgo call) and real workload throughput (H2D/D2H bandwidth, launch
   latency) — be honest in the writeup if cgo turns out faster for raw call
   overhead; the pitch is "no cgo needed," not necessarily "faster."
10. **Docs + v0.1.0 tag.** README with the pitch, the driver-api-coverage table, and
    a copy-pasteable vecadd quickstart.

Do not attempt reflection-based generic kernel-argument marshaling, multi-GPU
orchestration, or a Linux port before milestone 6 (kernel launch) works reliably.

## 6. Testing strategy

- All integration tests require an actual NVIDIA GPU + driver present — guard with
  a build tag or a runtime skip (`t.Skip` if `cuInit` fails) so CI without a GPU
  runner doesn't just fail red; make the failure mode explicit ("skipped: no CUDA
  device").
- If you get access to a self-hosted CI runner with a GPU, wire it in; otherwise
  document that CI only validates compilation, and real hardware testing happens
  locally on the RTX 4060 before tagging releases.
- Unit-test the `CUresult -> error` table and any argument-packing helpers without
  needing a GPU at all — pure logic, should run in vanilla CI.
- Memory safety tests: deliberately trigger double-free, use-after-free-on-Go-side
  scenarios in tests to confirm cugo returns errors rather than corrupting state
  (the driver API itself will often catch these, but confirm cugo surfaces it
  cleanly).

## 7. Documentation & README expectations

- One-paragraph pitch: no-cgo CUDA driver bindings, cross-compilation-friendly,
  runtime-only NVIDIA driver dependency.
- Comparison table: cugo vs `gorgonia/cu` (cgo-based) — build requirements, platform
  support, API coverage, maturity. Be honest that cugo is new and has far less API
  surface covered initially.
- Driver API coverage table (`docs/driver-api-coverage.md`) linked prominently —
  people evaluating a binding library want to know immediately whether the specific
  function they need is wrapped yet.
- Quickstart: the vecadd example, copy-pasteable, with the exact `nvcc` command
  used to regenerate the embedded PTX if someone wants to modify the kernel.
- Clear "status: experimental / pre-1.0, Windows-only for now" banner.

## 8. Explicit non-goals

- Not a CUDA compiler — never parses or compiles `.cu` source itself; always shells
  out to `nvcc`.
- Not cuDNN/cuBLAS/cuFFT bindings — driver API only, v1.
- Not a tensor/array library or ML framework layer — that would sit on top of cugo,
  not inside it.
- Not Linux-supported in v1 (architecture should make it additive later, but don't
  build it now).
- Not attempting the actual GPU kernel-mode driver — this wraps NVIDIA's existing
  user-mode driver API, full stop.

## 9. First commit checklist

- [ ] `go.mod`, module path, Go 1.22+
- [ ] `internal/nvapi` with `NewLazySystemDLL` setup + `cuInit`/`cuDeviceGetCount`
      bound and a passing milestone-1 test
- [ ] `docs/design.md` and `docs/decisions.md` stubs
- [ ] `docs/driver-api-coverage.md` stub (even if just 2 functions listed so far)
- [ ] README with pitch section filled in
- [ ] Apache-2.0 LICENSE
- [ ] `.github/workflows/ci.yml` — at minimum, compiles on windows-latest (GPU
      integration tests will skip without a GPU runner; note this in the workflow
      comments)

Start now with milestone 1. Do not write `driver/launch.go` or attempt the vecadd
example before `cuInit` + device enumeration is proven working on real hardware.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-09-08T16:24:30+03:30.

The user's current state is as follows:
Active Document: c:\Users\Daniel\Desktop\uringo\bench\comparison.md (LANGUAGE_MARKDOWN)
Cursor is on line: 1
Other open documents:
- c:\Users\Daniel\Desktop\uringo\conn\conn_test.go (LANGUAGE_GO)
- c:\Users\Daniel\Desktop\uringo\conn\listener.go (LANGUAGE_GO)
- c:\Users\Daniel\Desktop\uringo\conn\conn.go (LANGUAGE_GO)
- c:\Users\Daniel\Desktop\uringo\internal\syscall\syscall_linux.go (LANGUAGE_GO)
- c:\Users\Daniel\Desktop\uringo\internal\uapi\uapi_other.go (LANGUAGE_GO)
</ADDITIONAL_METADATA>
<USER_SETTINGS_CHANGE>
The user changed setting `Model Selection` from None to Gemini 3.8 Flash (Medium). No need to comment on this change if the user doesn't ask about it. If reporting what model you are, please use a human readable name instead of the exact string.
</USER_SETTINGS_CHANGE>