# CUDA Driver API Coverage

This document tracks the functions exposed by `cugo` vs the full NVIDIA CUDA Driver API surface.

| CUDA Driver Function | `internal/nvapi` | `driver` | Status | Notes |
|---|---|---|---|---|
| `cuInit` | Yes | `driver.Init` | Stable | Driver initialization |
| `cuDriverGetVersion` | Yes | `driver.DriverVersion` | Stable | Query driver version |
| `cuDeviceGetCount` | Yes | `driver.DeviceCount` | Stable | Query number of devices |
| `cuDeviceGet` | Yes | `driver.GetDevice` | Stable | Query device handle |
| `cuDeviceGetName` | Yes | `Device.Name` | Stable | Query device model name |
| `cuDeviceTotalMem_v2` | Yes | `Device.TotalMemory` | Stable | Query total memory |
| `cuDeviceGetAttribute` | Yes | `Device.Attribute`, `Device.ComputeCapability` | Stable | Query hardware attributes |
| `cuCtxCreate_v2` | Planned | `Device.CreateContext` | Planned | Milestone 2 |
| `cuCtxDestroy_v2` | Planned | `Context.Destroy` | Planned | Milestone 2 |
| `cuCtxSetCurrent` | Planned | `Context.SetCurrent` | Planned | Milestone 2 |
| `cuMemAlloc_v2` | Planned | `Context.Alloc` | Planned | Milestone 2 |
| `cuMemFree_v2` | Planned | `Context.Free` | Planned | Milestone 2 |
| `cuMemcpyHtoD_v2` | Planned | `Context.CopyHtoD` | Planned | Milestone 2 |
| `cuMemcpyDtoH_v2` | Planned | `Context.CopyDtoH` | Planned | Milestone 2 |
| `cuModuleLoadData` | Planned | `Context.LoadModuleData` | Planned | Milestone 3 |
| `cuModuleGetFunction` | Planned | `Module.Function` | Planned | Milestone 3 |
| `cuLaunchKernel` | Planned | `Function.Launch` | Planned | Milestone 4 |
| `cuStreamCreate` | Planned | `Context.CreateStream` | Planned | Milestone 5 |
| `cuStreamSynchronize` | Planned | `Stream.Synchronize` | Planned | Milestone 5 |
| `cuEventCreate` | Planned | `Context.CreateEvent` | Planned | Milestone 5 |
| `cuEventElapsedTime` | Planned | `Event.ElapsedTime` | Planned | Milestone 5 |
