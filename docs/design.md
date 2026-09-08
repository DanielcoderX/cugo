# cugo Design Document

## 1. Overview
`cugo` provides idiomatic Go bindings for the NVIDIA CUDA Driver API without `cgo`. It targets zero-setup builds: no C toolchain, no CUDA Toolkit headers needed at build time.

## 2. Layering Architecture

```
+-------------------------------------------------------------+
| Consumer Applications / Examples (e.g. vecadd, device-info) |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| package driver                                              |
| - Device, Context, DevicePtr, Module, Function, Stream, Event|
| - Idiomatic Go types, methods, error returns                |
| - Memory management & explicit kernel argument marshaling   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| package internal/nvapi                                      |
| - Raw ABI bindings (windows.NewLazySystemDLL / Proc.Call)   |
| - Exact CUDA Driver C ABI struct layouts & CUresult enums   |
| - Zero runtime dependencies beyond golang.org/x/sys         |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| nvcuda.dll (Windows User-Mode Driver)                       |
+-------------------------------------------------------------+
```

## 3. ABI Calling Conventions
- On `windows/amd64`, Go's `Proc.Call(args...)` complies with the Microsoft x64 64-bit fastcall convention.
- All pointer arguments (`*int32`, `*uint64`, `*CUdevice`) are passed as `uintptr(unsafe.Pointer(ptr))`.
- 64-bit device memory addresses (`CUdeviceptr`) map directly to `uintptr`.
- Handles (`CUdevice`, `CUcontext`, `CUmodule`, `CUfunction`, etc.) preserve pointer and integer semantics matching NVIDIA's `cuda.h`.

## 4. Error Handling Strategy
Every ABI function returns `CUresult` as its primary return value. `internal/nvapi.ResultToError` converts non-zero return values into descriptive Go `error` types with the exact enum identifier and description.
Missing DLL or missing proc entries return distinct errors (`ErrDriverNotFound`, `ErrProcNotFound`).
