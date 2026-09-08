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
