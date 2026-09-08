# CUDA Driver API Coverage

This document tracks the functions exposed by `cugo` vs the full NVIDIA CUDA Driver API surface.

| CUDA Driver Function | `internal/nvapi` | `driver` | Status | Notes |
|---|---|---|---|---|
| `cuInit` | Yes | `driver.Init` | Stable | Driver initialization |
| `cuDriverGetVersion` | Yes | `driver.DriverVersion` | Stable | Query driver version |
| `cuDeviceGetCount` | Yes | `driver.DeviceCount` | Stable | Query number of devices |
| `cuDeviceGet` | Yes | `driver.GetDevice` | Stable | Query device handle |
| `cuDeviceGetName` | Yes | `Device.Name` | Stable | Query device model name |
| `cuDeviceTotalMem_v2` | Yes | `Device.TotalMemory` | Stable | Query total global memory |
| `cuDeviceGetAttribute` | Yes | `Device.Attribute`, `Device.ComputeCapability` | Stable | Query hardware attributes (154 constants) |
| `cuCtxCreate_v2` | Yes | `Device.CreateContext` | Stable | Create context & set current |
| `cuCtxDestroy_v2` | Yes | `Context.Destroy` | Stable | Destroy context & cleanup |
| `cuCtxSetCurrent` | Yes | `Context.SetCurrent`, `Context.EnsureCurrent` | Stable | Bind context to current thread |
| `cuCtxGetCurrent` | Yes | `CurrentContext()` | Stable | Query current active context |
| `cuMemAlloc_v2` | Yes | `Context.Alloc` | Stable | Linear device memory allocation |
| `cuMemFree_v2` | Yes | `Context.Free` | Stable | Free device memory allocation |
| `cuMemAllocHost_v2` | Yes | `Context.AllocHost` | Stable | Page-locked host memory allocation |
| `cuMemFreeHost` | Yes | `HostMem.Free` | Stable | Free page-locked host memory |
| `cuMemHostGetDevicePointer_v2` | Yes | `HostMem.DevicePointer` | Stable | Zero-copy GPU mapping of host memory |
| `cuMemAllocManaged` | Yes | `Context.AllocManaged` | Stable | Unified Memory allocation (CPU + GPU shared) |
| `cuMemPrefetchAsync` | Yes | `ManagedMem.PrefetchToDevice`, `ManagedMem.PrefetchToCPU` | Stable | Asynchronous page prefetching |
| `cuMemAdvise` | Yes | `ManagedMem.Advise` | Stable | Unified Memory usage hints & advice |
| `cuMemcpyHtoD_v2` | Yes | `Context.CopyHtoD` | Stable | Synchronous host to device copy |
| `cuMemcpyDtoH_v2` | Yes | `Context.CopyDtoH` | Stable | Synchronous device to host copy |
| `cuMemcpyHtoDAsync_v2`| Yes | `Stream.CopyHtoDAsync` | Stable | Asynchronous stream host to device copy |
| `cuMemcpyDtoHAsync_v2`| Yes | `Stream.CopyDtoHAsync` | Stable | Asynchronous stream device to host copy |
| `cuModuleLoadData` | Yes | `Context.LoadModuleData` | Stable | Load PTX/cubin bytecode |
| `cuModuleUnload` | Yes | `Module.Unload` | Stable | Unload module from context |
| `cuModuleGetFunction` | Yes | `Module.Function` | Stable | Resolve kernel entry point symbol |
| `cuLaunchKernel` | Yes | `Function.Launch` | Stable | Kernel grid/block dispatch + typed args |
| `cuStreamCreate` | Yes | `Context.CreateStream` | Stable | Create asynchronous stream |
| `cuStreamDestroy_v2` | Yes | `Stream.Destroy` | Stable | Destroy asynchronous stream |
| `cuStreamSynchronize` | Yes | `Stream.Synchronize` | Stable | Block on stream completion |
| `cuEventCreate` | Yes | `Context.CreateEvent` | Stable | Create timing/sync event |
| `cuEventDestroy_v2` | Yes | `Event.Destroy` | Stable | Destroy event |
| `cuEventRecord` | Yes | `Event.Record` | Stable | Record event on stream |
| `cuEventSynchronize` | Yes | `Event.Synchronize` | Stable | Block on event completion |
| `cuEventElapsedTime` | Yes | `driver.ElapsedTime` | Stable | Compute elapsed ms between events |
| `cuDeviceCanAccessPeer` | Yes | `Device.CanAccessPeer` | Stable | Query peer-to-peer capability |
| `cuCtxEnablePeerAccess` | Yes | `Context.EnablePeerAccess` | Stable | Enable direct P2P access between GPUs |
| `cuCtxDisablePeerAccess` | Yes | `Context.DisablePeerAccess` | Stable | Disable direct P2P access |
| `cuMemcpyPeer` | Yes | `driver.CopyPeer` | Stable | Direct GPU-to-GPU memory copy |
| `cuMemcpyPeerAsync` | Yes | `driver.CopyPeerAsync` | Stable | Asynchronous GPU-to-GPU copy |
| `cuDeviceGetP2PAttribute` | Yes | `Device.P2PAttribute` | Stable | Query P2P link performance & attributes |
| `cuOccupancyMaxActiveBlocksPerMultiprocessor` | Yes | `Function.MaxActiveBlocksPerMultiprocessor` | Stable | Calculate active blocks per SM |
| `cuOccupancyMaxPotentialBlockSize` | Yes | `Function.SuggestBlockSize` | Stable | Auto-tune block and grid sizes |
| `cuGraphCreate` | Yes | `internal/nvapi.CuGraphCreate` | Stable | Create CUDA graph |
| `cuGraphDestroy` | Yes | `Graph.Destroy` | Stable | Destroy CUDA graph |
| `cuStreamBeginCapture_v2` | Yes | `Stream.BeginCapture` | Stable | Begin stream graph capture |
| `cuStreamEndCapture` | Yes | `Stream.EndCapture` | Stable | End capture and create graph |
| `cuStreamIsCapturing` | Yes | `Stream.IsCapturing` | Stable | Query stream capture state |
| `cuGraphInstantiate_v2` | Yes | `Graph.Instantiate` | Stable | Instantiate executable graph DAG |
| `cuGraphLaunch` | Yes | `GraphExec.Launch` | Stable | Replay graph on stream |
| `cuGraphExecDestroy` | Yes | `GraphExec.Destroy` | Stable | Destroy executable graph |
| `cuMemAllocAsync` | Yes | `Context.AllocAsync` | Stable | Stream-ordered async allocation |
| `cuMemFreeAsync` | Yes | `Context.FreeAsync` | Stable | Stream-ordered async free |
| `cuDeviceGetDefaultMemPool` | Yes | `Device.DefaultMemPool` | Stable | Query default memory pool |
| `cuMemPoolTrimTo` | Yes | `MemPool.TrimTo` | Stable | Trim unallocated memory in pool |
| `cuMemPoolGetAttribute` | Yes | `MemPool.ReservedMemCurrent`, etc. | Stable | Query memory pool metrics |
| `cuMemPoolSetAttribute` | Yes | `MemPool.SetReleaseThreshold` | Stable | Tune memory pool thresholds |
