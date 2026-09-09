//go:build windows || linux

package nvapi

import (
	"errors"
	"fmt"
	"unsafe"
)

var (
// Initialization & Device Management
	procInit               = newDriverProc("cuInit")
	procDriverGetVersion   = newDriverProc("cuDriverGetVersion")
	procDeviceGetCount     = newDriverProc("cuDeviceGetCount")
	procDeviceGet          = newDriverProc("cuDeviceGet")
	procDeviceGetName      = newDriverProc("cuDeviceGetName")
	procDeviceTotalMem     = newDriverProc("cuDeviceTotalMem_v2")
	procDeviceGetAttribute = newDriverProc("cuDeviceGetAttribute")

	// Context Management
	procCtxCreate      = newDriverProc("cuCtxCreate_v2")
	procCtxDestroy     = newDriverProc("cuCtxDestroy_v2")
	procCtxSetCurrent  = newDriverProc("cuCtxSetCurrent")
	procCtxGetCurrent  = newDriverProc("cuCtxGetCurrent")
	procCtxSynchronize = newDriverProc("cuCtxSynchronize")

	// Memory Management
	procMemAlloc        = newDriverProc("cuMemAlloc_v2")
	procMemFree         = newDriverProc("cuMemFree_v2")
	procMemcpyHtoD      = newDriverProc("cuMemcpyHtoD_v2")
	procMemcpyDtoH      = newDriverProc("cuMemcpyDtoH_v2")
	procMemcpyHtoDAsync         = newDriverProc("cuMemcpyHtoDAsync_v2")
	procMemcpyDtoHAsync         = newDriverProc("cuMemcpyDtoHAsync_v2")
	procMemAllocHost            = newDriverProc("cuMemAllocHost_v2")
	procMemFreeHost             = newDriverProc("cuMemFreeHost")
	procMemHostGetDevicePointer = newDriverProc("cuMemHostGetDevicePointer_v2")
	procMemAllocManaged         = newDriverProc("cuMemAllocManaged")
	procMemPrefetchAsync        = newDriverProc("cuMemPrefetchAsync")
	procMemAdvise               = newDriverProc("cuMemAdvise")
	procMemAllocPitch           = newDriverProc("cuMemAllocPitch_v2")
	procMemcpy2D                = newDriverProc("cuMemcpy2D_v2")
	procMemcpy2DAsync           = newDriverProc("cuMemcpy2DAsync_v2")
	procArrayCreate             = newDriverProc("cuArrayCreate_v2")
	procArrayDestroy            = newDriverProc("cuArrayDestroy")

	// Peer-to-Peer Access
	procDeviceCanAccessPeer   = newDriverProc("cuDeviceCanAccessPeer")
	procCtxEnablePeerAccess   = newDriverProc("cuCtxEnablePeerAccess")
	procCtxDisablePeerAccess  = newDriverProc("cuCtxDisablePeerAccess")
	procMemcpyPeer            = newDriverProc("cuMemcpyPeer")
	procMemcpyPeerAsync       = newDriverProc("cuMemcpyPeerAsync")
	procDeviceGetP2PAttribute = newDriverProc("cuDeviceGetP2PAttribute")

	// Module & Kernel Launch
	procModuleLoadData   = newDriverProc("cuModuleLoadData")
	procModuleUnload     = newDriverProc("cuModuleUnload")
	procModuleGetFunction = newDriverProc("cuModuleGetFunction")
	procLaunchKernel     = newDriverProc("cuLaunchKernel")

	// Stream Management
	procStreamCreate      = newDriverProc("cuStreamCreate")
	procStreamDestroy     = newDriverProc("cuStreamDestroy_v2")
	procStreamSynchronize = newDriverProc("cuStreamSynchronize")

	// Event Management
	procEventCreate      = newDriverProc("cuEventCreate")
	procEventDestroy     = newDriverProc("cuEventDestroy_v2")
	procEventRecord      = newDriverProc("cuEventRecord")
	procEventSynchronize = newDriverProc("cuEventSynchronize")
	procEventElapsedTime = newDriverProc("cuEventElapsedTime")

	// Occupancy Calculation
	procOccupancyMaxActiveBlocksPerMultiprocessor = newDriverProc("cuOccupancyMaxActiveBlocksPerMultiprocessor")
	procOccupancyMaxPotentialBlockSize           = newDriverProc("cuOccupancyMaxPotentialBlockSize")

	// CUDA Graph Management
	procGraphCreate        = newDriverProc("cuGraphCreate")
	procGraphDestroy       = newDriverProc("cuGraphDestroy")
	procStreamBeginCapture = newDriverProc("cuStreamBeginCapture_v2")
	procStreamEndCapture   = newDriverProc("cuStreamEndCapture")
	procStreamIsCapturing  = newDriverProc("cuStreamIsCapturing")
	procGraphInstantiate             = newDriverProc("cuGraphInstantiate_v2")
	procGraphLaunch                  = newDriverProc("cuGraphLaunch")
	procGraphExecDestroy             = newDriverProc("cuGraphExecDestroy")
	procGraphGetNodes                = newDriverProc("cuGraphGetNodes")
	procGraphNodeGetType             = newDriverProc("cuGraphNodeGetType")
	procGraphExecKernelNodeSetParams = newDriverProc("cuGraphExecKernelNodeSetParams")


	// Stream-Ordered Memory Allocator
	procMemAllocAsync       = newDriverProc("cuMemAllocAsync")
	procMemFreeAsync        = newDriverProc("cuMemFreeAsync")
	procMemPoolCreate       = newDriverProc("cuMemPoolCreate")
	procMemPoolDestroy      = newDriverProc("cuMemPoolDestroy")
	procMemPoolTrimTo       = newDriverProc("cuMemPoolTrimTo")
	procMemPoolSetAttribute = newDriverProc("cuMemPoolSetAttribute")
	procMemPoolGetAttribute = newDriverProc("cuMemPoolGetAttribute")
	procDeviceGetDefaultMemPool = newDriverProc("cuDeviceGetDefaultMemPool")

	// Dynamic JIT Linker
	procLinkCreate   = newDriverProc("cuLinkCreate_v2")
	procLinkAddData  = newDriverProc("cuLinkAddData_v2")
	procLinkComplete = newDriverProc("cuLinkComplete")
	procLinkDestroy  = newDriverProc("cuLinkDestroy")

	// Profiler Control
	procProfilerStart = newDriverProc("cuProfilerStart")
	procProfilerStop  = newDriverProc("cuProfilerStop")

	procTexObjectCreate  = newDriverProc("cuTexObjectCreate")
	procTexObjectDestroy = newDriverProc("cuTexObjectDestroy")
	procSurfObjectCreate = newDriverProc("cuSurfObjectCreate")
	procSurfObjectDestroy = newDriverProc("cuSurfObjectDestroy")

	// Memory Set & Stream Wait Event
	procMemsetD8Async   = newDriverProc("cuMemsetD8Async")
	procMemsetD32Async  = newDriverProc("cuMemsetD32Async")
	procStreamWaitEvent = newDriverProc("cuStreamWaitEvent")

)

// CuInit initializes the CUDA driver API. Must be called before any other driver functions.
func CuInit(flags uint32) error {
	if err := procInit.Find(); err != nil {
		return fmt.Errorf("%w: cuInit: %v", ErrProcNotFound, err)
	}
	r, _, _ := procInit.Call(uintptr(flags))
	return ResultToError(CUresult(r))
}

// CuDriverGetVersion returns the CUDA driver version.
func CuDriverGetVersion(version *int32) error {
	if err := procDriverGetVersion.Find(); err != nil {
		return fmt.Errorf("%w: cuDriverGetVersion: %v", ErrProcNotFound, err)
	}
	r, _, _ := procDriverGetVersion.Call(uintptr(unsafe.Pointer(version)))
	return ResultToError(CUresult(r))
}

// CuDeviceGetCount returns the number of compute-capable devices available.
func CuDeviceGetCount(count *int32) error {
	if err := procDeviceGetCount.Find(); err != nil {
		return fmt.Errorf("%w: cuDeviceGetCount: %v", ErrProcNotFound, err)
	}
	r, _, _ := procDeviceGetCount.Call(uintptr(unsafe.Pointer(count)))
	return ResultToError(CUresult(r))
}

// CuDeviceGet returns a handle to a compute device by ordinal index.
func CuDeviceGet(device *CUdevice, ordinal int32) error {
	if err := procDeviceGet.Find(); err != nil {
		return fmt.Errorf("%w: cuDeviceGet: %v", ErrProcNotFound, err)
	}
	r, _, _ := procDeviceGet.Call(uintptr(unsafe.Pointer(device)), uintptr(ordinal))
	return ResultToError(CUresult(r))
}

// CuDeviceGetName retrieves the identifier string for a device.
func CuDeviceGetName(name []byte, dev CUdevice) error {
	if len(name) == 0 {
		return nil
	}
	if err := procDeviceGetName.Find(); err != nil {
		return fmt.Errorf("%w: cuDeviceGetName: %v", ErrProcNotFound, err)
	}
	r, _, _ := procDeviceGetName.Call(
		uintptr(unsafe.Pointer(&name[0])),
		uintptr(len(name)),
		uintptr(dev),
	)
	return ResultToError(CUresult(r))
}

// CuDeviceTotalMem returns the total amount of memory available on the device in bytes.
func CuDeviceTotalMem(bytes *uint64, dev CUdevice) error {
	if err := procDeviceTotalMem.Find(); err != nil {
		return fmt.Errorf("%w: cuDeviceTotalMem_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procDeviceTotalMem.Call(uintptr(unsafe.Pointer(bytes)), uintptr(dev))
	return ResultToError(CUresult(r))
}

// CuDeviceGetAttribute returns information about the device attribute.
func CuDeviceGetAttribute(val *int32, attrib CUdevice_attribute, dev CUdevice) error {
	if err := procDeviceGetAttribute.Find(); err != nil {
		return fmt.Errorf("%w: cuDeviceGetAttribute: %v", ErrProcNotFound, err)
	}
	r, _, _ := procDeviceGetAttribute.Call(
		uintptr(unsafe.Pointer(val)),
		uintptr(attrib),
		uintptr(dev),
	)
	return ResultToError(CUresult(r))
}

// CuCtxCreate creates a new CUDA context and associates it with the calling thread.
func CuCtxCreate(pctx *CUcontext, flags uint32, dev CUdevice) error {
	if err := procCtxCreate.Find(); err != nil {
		return fmt.Errorf("%w: cuCtxCreate_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procCtxCreate.Call(
		uintptr(unsafe.Pointer(pctx)),
		uintptr(flags),
		uintptr(dev),
	)
	return ResultToError(CUresult(r))
}

// CuCtxDestroy destroys a CUDA context.
func CuCtxDestroy(ctx CUcontext) error {
	if err := procCtxDestroy.Find(); err != nil {
		return fmt.Errorf("%w: cuCtxDestroy_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procCtxDestroy.Call(uintptr(ctx))
	return ResultToError(CUresult(r))
}

// CuCtxSetCurrent binds the specified CUDA context to the calling CPU thread.
func CuCtxSetCurrent(ctx CUcontext) error {
	if err := procCtxSetCurrent.Find(); err != nil {
		return fmt.Errorf("%w: cuCtxSetCurrent: %v", ErrProcNotFound, err)
	}
	r, _, _ := procCtxSetCurrent.Call(uintptr(ctx))
	return ResultToError(CUresult(r))
}

// CuCtxGetCurrent returns the CUDA context bound to the calling CPU thread.
func CuCtxGetCurrent(pctx *CUcontext) error {
	if err := procCtxGetCurrent.Find(); err != nil {
		return fmt.Errorf("%w: cuCtxGetCurrent: %v", ErrProcNotFound, err)
	}
	r, _, _ := procCtxGetCurrent.Call(uintptr(unsafe.Pointer(pctx)))
	return ResultToError(CUresult(r))
}

// CuCtxSynchronize blocks until all tasks in the current context have completed.
func CuCtxSynchronize() error {
	if err := procCtxSynchronize.Find(); err != nil {
		return fmt.Errorf("%w: cuCtxSynchronize: %v", ErrProcNotFound, err)
	}
	r, _, _ := procCtxSynchronize.Call()
	return ResultToError(CUresult(r))
}

// CuMemAlloc allocates bytesize bytes of linear memory on the device.
func CuMemAlloc(dptr *CUdeviceptr, bytesize uint64) error {
	if err := procMemAlloc.Find(); err != nil {
		return fmt.Errorf("%w: cuMemAlloc_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemAlloc.Call(uintptr(unsafe.Pointer(dptr)), uintptr(bytesize))
	return ResultToError(CUresult(r))
}

// CuMemFree frees memory allocated on the device.
func CuMemFree(dptr CUdeviceptr) error {
	if err := procMemFree.Find(); err != nil {
		return fmt.Errorf("%w: cuMemFree_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemFree.Call(uintptr(dptr))
	return ResultToError(CUresult(r))
}

// CuMemcpyHtoD copies byteCount bytes from host memory to device memory.
func CuMemcpyHtoD(dstDevice CUdeviceptr, srcHost []byte) error {
	if len(srcHost) == 0 {
		return nil
	}
	if err := procMemcpyHtoD.Find(); err != nil {
		return fmt.Errorf("%w: cuMemcpyHtoD_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemcpyHtoD.Call(
		uintptr(dstDevice),
		uintptr(unsafe.Pointer(&srcHost[0])),
		uintptr(len(srcHost)),
	)
	return ResultToError(CUresult(r))
}

// CuMemcpyDtoH copies byteCount bytes from device memory to host memory.
func CuMemcpyDtoH(dstHost []byte, srcDevice CUdeviceptr) error {
	if len(dstHost) == 0 {
		return nil
	}
	if err := procMemcpyDtoH.Find(); err != nil {
		return fmt.Errorf("%w: cuMemcpyDtoH_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemcpyDtoH.Call(
		uintptr(unsafe.Pointer(&dstHost[0])),
		uintptr(srcDevice),
		uintptr(len(dstHost)),
	)
	return ResultToError(CUresult(r))
}

// CuMemcpyHtoDAsync copies memory from host to device asynchronously.
func CuMemcpyHtoDAsync(dstDevice CUdeviceptr, srcHost []byte, hStream CUstream) error {
	if len(srcHost) == 0 {
		return nil
	}
	if err := procMemcpyHtoDAsync.Find(); err != nil {
		return fmt.Errorf("%w: cuMemcpyHtoDAsync_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemcpyHtoDAsync.Call(
		uintptr(dstDevice),
		uintptr(unsafe.Pointer(&srcHost[0])),
		uintptr(len(srcHost)),
		uintptr(hStream),
	)
	return ResultToError(CUresult(r))
}

// CuMemcpyDtoHAsync copies memory from device to host asynchronously.
func CuMemcpyDtoHAsync(dstHost []byte, srcDevice CUdeviceptr, hStream CUstream) error {
	if len(dstHost) == 0 {
		return nil
	}
	if err := procMemcpyDtoHAsync.Find(); err != nil {
		return fmt.Errorf("%w: cuMemcpyDtoHAsync_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemcpyDtoHAsync.Call(
		uintptr(unsafe.Pointer(&dstHost[0])),
		uintptr(srcDevice),
		uintptr(len(dstHost)),
		uintptr(hStream),
	)
	return ResultToError(CUresult(r))
}

// CuMemAllocHost allocates page-locked (pinned) host memory.
func CuMemAllocHost(pp *unsafe.Pointer, bytesize uint64) error {
	if err := procMemAllocHost.Find(); err != nil {
		return fmt.Errorf("%w: cuMemAllocHost_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemAllocHost.Call(uintptr(unsafe.Pointer(pp)), uintptr(bytesize))
	return ResultToError(CUresult(r))
}

// CuMemFreeHost frees page-locked host memory.
func CuMemFreeHost(p unsafe.Pointer) error {
	if err := procMemFreeHost.Find(); err != nil {
		return fmt.Errorf("%w: cuMemFreeHost: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemFreeHost.Call(uintptr(p))
	return ResultToError(CUresult(r))
}

// CuMemHostGetDevicePointer returns a device pointer corresponding to mapped pinned host memory.
func CuMemHostGetDevicePointer(pdptr *CUdeviceptr, p unsafe.Pointer, flags uint32) error {
	if err := procMemHostGetDevicePointer.Find(); err != nil {
		return fmt.Errorf("%w: cuMemHostGetDevicePointer_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemHostGetDevicePointer.Call(
		uintptr(unsafe.Pointer(pdptr)),
		uintptr(p),
		uintptr(flags),
	)
	return ResultToError(CUresult(r))
}

// CuMemAllocManaged allocates memory managed automatically by Unified Memory.
func CuMemAllocManaged(dptr *CUdeviceptr, bytesize uint64, flags uint32) error {
	if err := procMemAllocManaged.Find(); err != nil {
		return fmt.Errorf("%w: cuMemAllocManaged: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemAllocManaged.Call(
		uintptr(unsafe.Pointer(dptr)),
		uintptr(bytesize),
		uintptr(flags),
	)
	return ResultToError(CUresult(r))
}

// CuMemPrefetchAsync prefetches unified memory to the destination device or CPU.
func CuMemPrefetchAsync(devPtr CUdeviceptr, count uint64, dstDevice CUdevice, hStream CUstream) error {
	if err := procMemPrefetchAsync.Find(); err != nil {
		return fmt.Errorf("%w: cuMemPrefetchAsync: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemPrefetchAsync.Call(
		uintptr(devPtr),
		uintptr(count),
		uintptr(dstDevice),
		uintptr(hStream),
	)
	return ResultToError(CUresult(r))
}

// CuMemAdvise advises the Unified Memory subsystem about usage patterns for memory ranges.
func CuMemAdvise(devPtr CUdeviceptr, count uint64, advice CUmem_advise, device CUdevice) error {
	if err := procMemAdvise.Find(); err != nil {
		return fmt.Errorf("%w: cuMemAdvise: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemAdvise.Call(
		uintptr(devPtr),
		uintptr(count),
		uintptr(advice),
		uintptr(device),
	)
	return ResultToError(CUresult(r))
}

// CuDeviceCanAccessPeer queries if the device can directly access peerDevice memory.
func CuDeviceCanAccessPeer(canAccess *int32, dev CUdevice, peerDev CUdevice) error {
	if err := procDeviceCanAccessPeer.Find(); err != nil {
		return fmt.Errorf("%w: cuDeviceCanAccessPeer: %v", ErrProcNotFound, err)
	}
	r, _, _ := procDeviceCanAccessPeer.Call(
		uintptr(unsafe.Pointer(canAccess)),
		uintptr(dev),
		uintptr(peerDev),
	)
	return ResultToError(CUresult(r))
}

// CuCtxEnablePeerAccess enables direct peer-to-peer access to memory allocated in peerContext.
func CuCtxEnablePeerAccess(peerContext CUcontext, flags uint32) error {
	if err := procCtxEnablePeerAccess.Find(); err != nil {
		return fmt.Errorf("%w: cuCtxEnablePeerAccess: %v", ErrProcNotFound, err)
	}
	r, _, _ := procCtxEnablePeerAccess.Call(
		uintptr(peerContext),
		uintptr(flags),
	)
	return ResultToError(CUresult(r))
}

// CuCtxDisablePeerAccess disables direct peer-to-peer access to memory allocated in peerContext.
func CuCtxDisablePeerAccess(peerContext CUcontext) error {
	if err := procCtxDisablePeerAccess.Find(); err != nil {
		return fmt.Errorf("%w: cuCtxDisablePeerAccess: %v", ErrProcNotFound, err)
	}
	r, _, _ := procCtxDisablePeerAccess.Call(uintptr(peerContext))
	return ResultToError(CUresult(r))
}

// CuMemcpyPeer copies memory between two contexts directly.
func CuMemcpyPeer(
	dstDevice CUdeviceptr,
	dstContext CUcontext,
	srcDevice CUdeviceptr,
	srcContext CUcontext,
	byteCount uint64,
) error {
	if err := procMemcpyPeer.Find(); err != nil {
		return fmt.Errorf("%w: cuMemcpyPeer: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemcpyPeer.Call(
		uintptr(dstDevice),
		uintptr(dstContext),
		uintptr(srcDevice),
		uintptr(srcContext),
		uintptr(byteCount),
	)
	return ResultToError(CUresult(r))
}

// CuMemcpyPeerAsync copies memory between two contexts asynchronously.
func CuMemcpyPeerAsync(
	dstDevice CUdeviceptr,
	dstContext CUcontext,
	srcDevice CUdeviceptr,
	srcContext CUcontext,
	byteCount uint64,
	hStream CUstream,
) error {
	if err := procMemcpyPeerAsync.Find(); err != nil {
		return fmt.Errorf("%w: cuMemcpyPeerAsync: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemcpyPeerAsync.Call(
		uintptr(dstDevice),
		uintptr(dstContext),
		uintptr(srcDevice),
		uintptr(srcContext),
		uintptr(byteCount),
		uintptr(hStream),
	)
	return ResultToError(CUresult(r))
}

// CuDeviceGetP2PAttribute queries attributes of the P2P connection between two devices.
func CuDeviceGetP2PAttribute(value *int32, attrib CUdevice_P2PAttribute, srcDevice CUdevice, dstDevice CUdevice) error {
	if err := procDeviceGetP2PAttribute.Find(); err != nil {
		return fmt.Errorf("%w: cuDeviceGetP2PAttribute: %v", ErrProcNotFound, err)
	}
	r, _, _ := procDeviceGetP2PAttribute.Call(
		uintptr(unsafe.Pointer(value)),
		uintptr(attrib),
		uintptr(srcDevice),
		uintptr(dstDevice),
	)
	return ResultToError(CUresult(r))
}

// CuModuleLoadData loads a compute module from raw PTX or cubin data.
func CuModuleLoadData(module *CUmodule, image []byte) error {
	if len(image) == 0 {
		return errors.New("cugo: empty module image")
	}
	if err := procModuleLoadData.Find(); err != nil {
		return fmt.Errorf("%w: cuModuleLoadData: %v", ErrProcNotFound, err)
	}
	r, _, _ := procModuleLoadData.Call(
		uintptr(unsafe.Pointer(module)),
		uintptr(unsafe.Pointer(&image[0])),
	)
	return ResultToError(CUresult(r))
}

// CuModuleUnload unloads a module from the current CUDA context.
func CuModuleUnload(module CUmodule) error {
	if err := procModuleUnload.Find(); err != nil {
		return fmt.Errorf("%w: cuModuleUnload: %v", ErrProcNotFound, err)
	}
	r, _, _ := procModuleUnload.Call(uintptr(module))
	return ResultToError(CUresult(r))
}

// CuModuleGetFunction returns a function handle from a loaded module.
func CuModuleGetFunction(hfunc *CUfunction, hmod CUmodule, name string) error {
	if err := procModuleGetFunction.Find(); err != nil {
		return fmt.Errorf("%w: cuModuleGetFunction: %v", ErrProcNotFound, err)
	}
	cName := append([]byte(name), 0)
	r, _, _ := procModuleGetFunction.Call(
		uintptr(unsafe.Pointer(hfunc)),
		uintptr(hmod),
		uintptr(unsafe.Pointer(&cName[0])),
	)
	return ResultToError(CUresult(r))
}

// CuLaunchKernel launches a CUDA kernel on the device.
func CuLaunchKernel(
	f CUfunction,
	gridDimX, gridDimY, gridDimZ uint32,
	blockDimX, blockDimY, blockDimZ uint32,
	sharedMemBytes uint32,
	hStream CUstream,
	kernelParams unsafe.Pointer,
	extra unsafe.Pointer,
) error {
	if err := procLaunchKernel.Find(); err != nil {
		return fmt.Errorf("%w: cuLaunchKernel: %v", ErrProcNotFound, err)
	}
	r, _, _ := procLaunchKernel.Call(
		uintptr(f),
		uintptr(gridDimX),
		uintptr(gridDimY),
		uintptr(gridDimZ),
		uintptr(blockDimX),
		uintptr(blockDimY),
		uintptr(blockDimZ),
		uintptr(sharedMemBytes),
		uintptr(hStream),
		uintptr(kernelParams),
		uintptr(extra),
	)
	return ResultToError(CUresult(r))
}

// CuStreamCreate creates an asynchronous stream.
func CuStreamCreate(phStream *CUstream, flags uint32) error {
	if err := procStreamCreate.Find(); err != nil {
		return fmt.Errorf("%w: cuStreamCreate: %v", ErrProcNotFound, err)
	}
	r, _, _ := procStreamCreate.Call(
		uintptr(unsafe.Pointer(phStream)),
		uintptr(flags),
	)
	return ResultToError(CUresult(r))
}

// CuStreamDestroy destroys an asynchronous stream.
func CuStreamDestroy(hStream CUstream) error {
	if err := procStreamDestroy.Find(); err != nil {
		return fmt.Errorf("%w: cuStreamDestroy_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procStreamDestroy.Call(uintptr(hStream))
	return ResultToError(CUresult(r))
}

// CuStreamSynchronize waits for stream tasks to complete.
func CuStreamSynchronize(hStream CUstream) error {
	if err := procStreamSynchronize.Find(); err != nil {
		return fmt.Errorf("%w: cuStreamSynchronize: %v", ErrProcNotFound, err)
	}
	r, _, _ := procStreamSynchronize.Call(uintptr(hStream))
	return ResultToError(CUresult(r))
}

// CuEventCreate creates an event.
func CuEventCreate(phEvent *CUevent, flags uint32) error {
	if err := procEventCreate.Find(); err != nil {
		return fmt.Errorf("%w: cuEventCreate: %v", ErrProcNotFound, err)
	}
	r, _, _ := procEventCreate.Call(
		uintptr(unsafe.Pointer(phEvent)),
		uintptr(flags),
	)
	return ResultToError(CUresult(r))
}

// CuEventDestroy destroys an event.
func CuEventDestroy(hEvent CUevent) error {
	if err := procEventDestroy.Find(); err != nil {
		return fmt.Errorf("%w: cuEventDestroy_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procEventDestroy.Call(uintptr(hEvent))
	return ResultToError(CUresult(r))
}

// CuEventRecord records an event on the specified stream.
func CuEventRecord(hEvent CUevent, hStream CUstream) error {
	if err := procEventRecord.Find(); err != nil {
		return fmt.Errorf("%w: cuEventRecord: %v", ErrProcNotFound, err)
	}
	r, _, _ := procEventRecord.Call(uintptr(hEvent), uintptr(hStream))
	return ResultToError(CUresult(r))
}

// CuEventSynchronize waits until the completion of an event.
func CuEventSynchronize(hEvent CUevent) error {
	if err := procEventSynchronize.Find(); err != nil {
		return fmt.Errorf("%w: cuEventSynchronize: %v", ErrProcNotFound, err)
	}
	r, _, _ := procEventSynchronize.Call(uintptr(hEvent))
	return ResultToError(CUresult(r))
}

// CuEventElapsedTime computes the elapsed time between two events in milliseconds.
func CuEventElapsedTime(milliseconds *float32, hStart CUevent, hEnd CUevent) error {
	if err := procEventElapsedTime.Find(); err != nil {
		return fmt.Errorf("%w: cuEventElapsedTime: %v", ErrProcNotFound, err)
	}
	r, _, _ := procEventElapsedTime.Call(
		uintptr(unsafe.Pointer(milliseconds)),
		uintptr(hStart),
		uintptr(hEnd),
	)
	return ResultToError(CUresult(r))
}

// CuOccupancyMaxActiveBlocksPerMultiprocessor returns the maximum active blocks per multiprocessor for a given kernel and block size.
func CuOccupancyMaxActiveBlocksPerMultiprocessor(
	numBlocks *int32,
	fn CUfunction,
	blockSize int32,
	dynamicSMemSize uint64,
) error {
	if err := procOccupancyMaxActiveBlocksPerMultiprocessor.Find(); err != nil {
		return fmt.Errorf("%w: cuOccupancyMaxActiveBlocksPerMultiprocessor: %v", ErrProcNotFound, err)
	}
	r, _, _ := procOccupancyMaxActiveBlocksPerMultiprocessor.Call(
		uintptr(unsafe.Pointer(numBlocks)),
		uintptr(fn),
		uintptr(blockSize),
		uintptr(dynamicSMemSize),
	)
	return ResultToError(CUresult(r))
}

// CuOccupancyMaxPotentialBlockSize suggests a block size that yields maximum occupancy.
func CuOccupancyMaxPotentialBlockSize(
	minGridSize *int32,
	blockSize *int32,
	fn CUfunction,
	blockSizeToDynamicSMemSize uintptr,
	dynamicSMemSize uint64,
	blockSizeLimit int32,
) error {
	if err := procOccupancyMaxPotentialBlockSize.Find(); err != nil {
		return fmt.Errorf("%w: cuOccupancyMaxPotentialBlockSize: %v", ErrProcNotFound, err)
	}
	r, _, _ := procOccupancyMaxPotentialBlockSize.Call(
		uintptr(unsafe.Pointer(minGridSize)),
		uintptr(unsafe.Pointer(blockSize)),
		uintptr(fn),
		blockSizeToDynamicSMemSize,
		uintptr(dynamicSMemSize),
		uintptr(blockSizeLimit),
	)
	return ResultToError(CUresult(r))
}

// CuGraphCreate creates a new CUDA graph.
func CuGraphCreate(phGraph *CUgraph, flags uint32) error {
	if err := procGraphCreate.Find(); err != nil {
		return fmt.Errorf("%w: cuGraphCreate: %v", ErrProcNotFound, err)
	}
	r, _, _ := procGraphCreate.Call(uintptr(unsafe.Pointer(phGraph)), uintptr(flags))
	return ResultToError(CUresult(r))
}

// CuGraphDestroy destroys a CUDA graph.
func CuGraphDestroy(hGraph CUgraph) error {
	if err := procGraphDestroy.Find(); err != nil {
		return fmt.Errorf("%w: cuGraphDestroy: %v", ErrProcNotFound, err)
	}
	r, _, _ := procGraphDestroy.Call(uintptr(hGraph))
	return ResultToError(CUresult(r))
}

// CuStreamBeginCapture begins graph stream capture on the specified stream.
func CuStreamBeginCapture(hStream CUstream, mode CUstreamCaptureMode) error {
	if err := procStreamBeginCapture.Find(); err != nil {
		return fmt.Errorf("%w: cuStreamBeginCapture_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procStreamBeginCapture.Call(uintptr(hStream), uintptr(mode))
	return ResultToError(CUresult(r))
}

// CuStreamEndCapture ends graph stream capture and returns the captured graph.
func CuStreamEndCapture(hStream CUstream, phGraph *CUgraph) error {
	if err := procStreamEndCapture.Find(); err != nil {
		return fmt.Errorf("%w: cuStreamEndCapture: %v", ErrProcNotFound, err)
	}
	r, _, _ := procStreamEndCapture.Call(uintptr(hStream), uintptr(unsafe.Pointer(phGraph)))
	return ResultToError(CUresult(r))
}

// CuStreamIsCapturing returns whether a stream is currently capturing.
func CuStreamIsCapturing(hStream CUstream, captureStatus *CUstreamCaptureStatus) error {
	if err := procStreamIsCapturing.Find(); err != nil {
		return fmt.Errorf("%w: cuStreamIsCapturing: %v", ErrProcNotFound, err)
	}
	r, _, _ := procStreamIsCapturing.Call(uintptr(hStream), uintptr(unsafe.Pointer(captureStatus)))
	return ResultToError(CUresult(r))
}

// CuGraphInstantiate instantiates an executable graph from a graph template.
func CuGraphInstantiate(phGraphExec *CUgraphExec, hGraph CUgraph) error {
	if err := procGraphInstantiate.Find(); err != nil {
		return fmt.Errorf("%w: cuGraphInstantiate_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procGraphInstantiate.Call(
		uintptr(unsafe.Pointer(phGraphExec)),
		uintptr(hGraph),
		0,
		0,
		0,
	)
	return ResultToError(CUresult(r))
}

// CuGraphLaunch executes an instantiated graph on a stream.
func CuGraphLaunch(hGraphExec CUgraphExec, hStream CUstream) error {
	if err := procGraphLaunch.Find(); err != nil {
		return fmt.Errorf("%w: cuGraphLaunch: %v", ErrProcNotFound, err)
	}
	r, _, _ := procGraphLaunch.Call(uintptr(hGraphExec), uintptr(hStream))
	return ResultToError(CUresult(r))
}

// CuGraphExecDestroy destroys an executable graph.
func CuGraphExecDestroy(hGraphExec CUgraphExec) error {
	if err := procGraphExecDestroy.Find(); err != nil {
		return fmt.Errorf("%w: cuGraphExecDestroy: %v", ErrProcNotFound, err)
	}
	r, _, _ := procGraphExecDestroy.Call(uintptr(hGraphExec))
	return ResultToError(CUresult(r))
}

// CuGraphGetNodes returns the nodes in a graph.
// If nodes is nil, numNodes returns the total number of nodes in hGraph.
func CuGraphGetNodes(hGraph CUgraph, nodes *CUgraphNode, numNodes *uint64) error {
	if err := procGraphGetNodes.Find(); err != nil {
		return fmt.Errorf("%w: cuGraphGetNodes: %v", ErrProcNotFound, err)
	}
	r, _, _ := procGraphGetNodes.Call(
		uintptr(hGraph),
		uintptr(unsafe.Pointer(nodes)),
		uintptr(unsafe.Pointer(numNodes)),
	)
	return ResultToError(CUresult(r))
}

// CuGraphNodeGetType returns the type of a graph node.
func CuGraphNodeGetType(hNode CUgraphNode, nodeType *CUgraphNodeType) error {
	if err := procGraphNodeGetType.Find(); err != nil {
		return fmt.Errorf("%w: cuGraphNodeGetType: %v", ErrProcNotFound, err)
	}
	r, _, _ := procGraphNodeGetType.Call(
		uintptr(hNode),
		uintptr(unsafe.Pointer(nodeType)),
	)
	return ResultToError(CUresult(r))
}

// CuGraphExecKernelNodeSetParams updates the parameters of a kernel node in an instantiated graph.
func CuGraphExecKernelNodeSetParams(hGraphExec CUgraphExec, hNode CUgraphNode, nodeParams *CUDA_KERNEL_NODE_PARAMS) error {
	if err := procGraphExecKernelNodeSetParams.Find(); err != nil {
		return fmt.Errorf("%w: cuGraphExecKernelNodeSetParams: %v", ErrProcNotFound, err)
	}
	r, _, _ := procGraphExecKernelNodeSetParams.Call(
		uintptr(hGraphExec),
		uintptr(hNode),
		uintptr(unsafe.Pointer(nodeParams)),
	)
	return ResultToError(CUresult(r))
}


// CuMemAllocAsync allocates memory asynchronously on a stream.
func CuMemAllocAsync(dptr *CUdeviceptr, bytesize uint64, hStream CUstream) error {
	if err := procMemAllocAsync.Find(); err != nil {
		return fmt.Errorf("%w: cuMemAllocAsync: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemAllocAsync.Call(
		uintptr(unsafe.Pointer(dptr)),
		uintptr(bytesize),
		uintptr(hStream),
	)
	return ResultToError(CUresult(r))
}

// CuMemFreeAsync frees memory asynchronously on a stream.
func CuMemFreeAsync(dptr CUdeviceptr, hStream CUstream) error {
	if err := procMemFreeAsync.Find(); err != nil {
		return fmt.Errorf("%w: cuMemFreeAsync: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemFreeAsync.Call(uintptr(dptr), uintptr(hStream))
	return ResultToError(CUresult(r))
}

// CuDeviceGetDefaultMemPool returns the default memory pool for the specified device.
func CuDeviceGetDefaultMemPool(pool *CUmemoryPool, dev CUdevice) error {
	if err := procDeviceGetDefaultMemPool.Find(); err != nil {
		return fmt.Errorf("%w: cuDeviceGetDefaultMemPool: %v", ErrProcNotFound, err)
	}
	r, _, _ := procDeviceGetDefaultMemPool.Call(uintptr(unsafe.Pointer(pool)), uintptr(dev))
	return ResultToError(CUresult(r))
}

// CuMemPoolDestroy destroys a memory pool.
func CuMemPoolDestroy(pool CUmemoryPool) error {
	if err := procMemPoolDestroy.Find(); err != nil {
		return fmt.Errorf("%w: cuMemPoolDestroy: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemPoolDestroy.Call(uintptr(pool))
	return ResultToError(CUresult(r))
}

// CuMemPoolTrimTo trims memory pool reservations down to minBytesToKeep.
func CuMemPoolTrimTo(pool CUmemoryPool, minBytesToKeep uint64) error {
	if err := procMemPoolTrimTo.Find(); err != nil {
		return fmt.Errorf("%w: cuMemPoolTrimTo: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemPoolTrimTo.Call(uintptr(pool), uintptr(minBytesToKeep))
	return ResultToError(CUresult(r))
}

// CuMemPoolSetAttribute sets an attribute on a memory pool.
func CuMemPoolSetAttribute(pool CUmemoryPool, attr CUmemPool_attribute, value unsafe.Pointer) error {
	if err := procMemPoolSetAttribute.Find(); err != nil {
		return fmt.Errorf("%w: cuMemPoolSetAttribute: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemPoolSetAttribute.Call(uintptr(pool), uintptr(attr), uintptr(value))
	return ResultToError(CUresult(r))
}

// CuMemPoolGetAttribute queries an attribute from a memory pool.
func CuMemPoolGetAttribute(pool CUmemoryPool, attr CUmemPool_attribute, value unsafe.Pointer) error {
	if err := procMemPoolGetAttribute.Find(); err != nil {
		return fmt.Errorf("%w: cuMemPoolGetAttribute: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemPoolGetAttribute.Call(uintptr(pool), uintptr(attr), uintptr(value))
	return ResultToError(CUresult(r))
}

// CuLinkCreate creates a pending JIT linker invocation state.
func CuLinkCreate(
	numOptions uint32,
	options *CUjit_option,
	optionValues *unsafe.Pointer,
	stateOut *CUlinkState,
) error {
	if err := procLinkCreate.Find(); err != nil {
		return fmt.Errorf("%w: cuLinkCreate_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procLinkCreate.Call(
		uintptr(numOptions),
		uintptr(unsafe.Pointer(options)),
		uintptr(unsafe.Pointer(optionValues)),
		uintptr(unsafe.Pointer(stateOut)),
	)
	return ResultToError(CUresult(r))
}

// CuLinkAddData adds an input (e.g. PTX or Cubin bytes) to a pending linker invocation.
func CuLinkAddData(
	state CUlinkState,
	inputType CUjitInputType,
	data []byte,
	name string,
	numOptions uint32,
	options *CUjit_option,
	optionValues *unsafe.Pointer,
) error {
	if len(data) == 0 {
		return errors.New("cugo: empty link input data")
	}
	if err := procLinkAddData.Find(); err != nil {
		return fmt.Errorf("%w: cuLinkAddData_v2: %v", ErrProcNotFound, err)
	}
	var pName uintptr
	if len(name) > 0 {
		cName := append([]byte(name), 0)
		pName = uintptr(unsafe.Pointer(&cName[0]))
	}
	r, _, _ := procLinkAddData.Call(
		uintptr(state),
		uintptr(inputType),
		uintptr(unsafe.Pointer(&data[0])),
		uintptr(len(data)),
		pName,
		uintptr(numOptions),
		uintptr(unsafe.Pointer(options)),
		uintptr(unsafe.Pointer(optionValues)),
	)
	return ResultToError(CUresult(r))
}

// CuLinkComplete completes the pending linker action and returns the cubin image.
func CuLinkComplete(state CUlinkState, cubinOut *unsafe.Pointer, sizeOut *uint64) error {
	if err := procLinkComplete.Find(); err != nil {
		return fmt.Errorf("%w: cuLinkComplete: %v", ErrProcNotFound, err)
	}
	r, _, _ := procLinkComplete.Call(
		uintptr(state),
		uintptr(unsafe.Pointer(cubinOut)),
		uintptr(unsafe.Pointer(sizeOut)),
	)
	return ResultToError(CUresult(r))
}

// CuLinkDestroy destroys state for a JIT linker invocation.
func CuLinkDestroy(state CUlinkState) error {
	if err := procLinkDestroy.Find(); err != nil {
		return fmt.Errorf("%w: cuLinkDestroy: %v", ErrProcNotFound, err)
	}
	r, _, _ := procLinkDestroy.Call(uintptr(state))
	return ResultToError(CUresult(r))
}

// CuMemAllocPitch allocates pitched device memory.
func CuMemAllocPitch(
	dptr *CUdeviceptr,
	pPitch *uint64,
	widthInBytes uint64,
	height uint64,
	elementSizeBytes uint32,
) error {
	if err := procMemAllocPitch.Find(); err != nil {
		return fmt.Errorf("%w: cuMemAllocPitch_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemAllocPitch.Call(
		uintptr(unsafe.Pointer(dptr)),
		uintptr(unsafe.Pointer(pPitch)),
		uintptr(widthInBytes),
		uintptr(height),
		uintptr(elementSizeBytes),
	)
	return ResultToError(CUresult(r))
}

// CuMemcpy2D performs a 2D memory copy between host, device, and array memory.
func CuMemcpy2D(pCopy *CUDA_MEMCPY2D) error {
	if pCopy == nil {
		return errors.New("cugo: nil 2D copy params")
	}
	if err := procMemcpy2D.Find(); err != nil {
		return fmt.Errorf("%w: cuMemcpy2D_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemcpy2D.Call(uintptr(unsafe.Pointer(pCopy)))
	return ResultToError(CUresult(r))
}

// CuMemcpy2DAsync performs an asynchronous 2D memory copy on a stream.
func CuMemcpy2DAsync(pCopy *CUDA_MEMCPY2D, hStream CUstream) error {
	if pCopy == nil {
		return errors.New("cugo: nil 2D copy params")
	}
	if err := procMemcpy2DAsync.Find(); err != nil {
		return fmt.Errorf("%w: cuMemcpy2DAsync_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemcpy2DAsync.Call(uintptr(unsafe.Pointer(pCopy)), uintptr(hStream))
	return ResultToError(CUresult(r))
}

// CuArrayCreate creates a 2D CUDA array.
func CuArrayCreate(pHandle *CUarray, pAllocateArray *CUDA_ARRAY_DESCRIPTOR) error {
	if pAllocateArray == nil {
		return errors.New("cugo: nil array descriptor")
	}
	if err := procArrayCreate.Find(); err != nil {
		return fmt.Errorf("%w: cuArrayCreate_v2: %v", ErrProcNotFound, err)
	}
	r, _, _ := procArrayCreate.Call(
		uintptr(unsafe.Pointer(pHandle)),
		uintptr(unsafe.Pointer(pAllocateArray)),
	)
	return ResultToError(CUresult(r))
}

// CuArrayDestroy destroys a CUDA array.
func CuArrayDestroy(hArray CUarray) error {
	if err := procArrayDestroy.Find(); err != nil {
		return fmt.Errorf("%w: cuArrayDestroy: %v", ErrProcNotFound, err)
	}
	r, _, _ := procArrayDestroy.Call(uintptr(hArray))
	return ResultToError(CUresult(r))
}

// CuProfilerStart starts profiling data collection.
func CuProfilerStart() error {
	if err := procProfilerStart.Find(); err != nil {
		return fmt.Errorf("%w: cuProfilerStart: %v", ErrProcNotFound, err)
	}
	r, _, _ := procProfilerStart.Call()
	return ResultToError(CUresult(r))
}

// CuProfilerStop stops profiling data collection.
func CuProfilerStop() error {
	if err := procProfilerStop.Find(); err != nil {
		return fmt.Errorf("%w: cuProfilerStop: %v", ErrProcNotFound, err)
	}
	r, _, _ := procProfilerStop.Call()
	return ResultToError(CUresult(r))
}

// CuTexObjectCreate creates a CUDA texture object.
func CuTexObjectCreate(
	pTexObject *CUtexObject,
	pResDesc *CUDA_RESOURCE_DESC,
	pTexDesc *CUDA_TEXTURE_DESC,
	pResViewDesc unsafe.Pointer,
) error {
	if pTexObject == nil || pResDesc == nil || pTexDesc == nil {
		return errors.New("cugo: nil texture object parameter")
	}
	if err := procTexObjectCreate.Find(); err != nil {
		return fmt.Errorf("%w: cuTexObjectCreate: %v", ErrProcNotFound, err)
	}
	r, _, _ := procTexObjectCreate.Call(
		uintptr(unsafe.Pointer(pTexObject)),
		uintptr(unsafe.Pointer(pResDesc)),
		uintptr(unsafe.Pointer(pTexDesc)),
		uintptr(pResViewDesc),
	)
	return ResultToError(CUresult(r))
}

// CuTexObjectDestroy destroys a CUDA texture object.
func CuTexObjectDestroy(texObject CUtexObject) error {
	if err := procTexObjectDestroy.Find(); err != nil {
		return fmt.Errorf("%w: cuTexObjectDestroy: %v", ErrProcNotFound, err)
	}
	r, _, _ := procTexObjectDestroy.Call(uintptr(texObject))
	return ResultToError(CUresult(r))
}

// CuSurfObjectCreate creates a CUDA surface object.
func CuSurfObjectCreate(pSurfObject *CUsurfObject, pResDesc *CUDA_RESOURCE_DESC) error {
	if pSurfObject == nil || pResDesc == nil {
		return errors.New("cugo: nil surface object parameter")
	}
	if err := procSurfObjectCreate.Find(); err != nil {
		return fmt.Errorf("%w: cuSurfObjectCreate: %v", ErrProcNotFound, err)
	}
	r, _, _ := procSurfObjectCreate.Call(
		uintptr(unsafe.Pointer(pSurfObject)),
		uintptr(unsafe.Pointer(pResDesc)),
	)
	return ResultToError(CUresult(r))
}

// CuSurfObjectDestroy destroys a CUDA surface object.
func CuSurfObjectDestroy(surfObject CUsurfObject) error {
	if err := procSurfObjectDestroy.Find(); err != nil {
		return fmt.Errorf("%w: cuSurfObjectDestroy: %v", ErrProcNotFound, err)
	}
	r, _, _ := procSurfObjectDestroy.Call(uintptr(surfObject))
	return ResultToError(CUresult(r))
}

// CuMemsetD8Async sets device memory to an 8-bit value asynchronously on a stream.
func CuMemsetD8Async(dstDevice CUdeviceptr, uc uint8, N uint64, hStream CUstream) error {
	if err := procMemsetD8Async.Find(); err != nil {
		return fmt.Errorf("%w: cuMemsetD8Async: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemsetD8Async.Call(
		uintptr(dstDevice),
		uintptr(uc),
		uintptr(N),
		uintptr(hStream),
	)
	return ResultToError(CUresult(r))
}

// CuMemsetD32Async sets device memory to a 32-bit value asynchronously on a stream.
func CuMemsetD32Async(dstDevice CUdeviceptr, ui uint32, N uint64, hStream CUstream) error {
	if err := procMemsetD32Async.Find(); err != nil {
		return fmt.Errorf("%w: cuMemsetD32Async: %v", ErrProcNotFound, err)
	}
	r, _, _ := procMemsetD32Async.Call(
		uintptr(dstDevice),
		uintptr(ui),
		uintptr(N),
		uintptr(hStream),
	)
	return ResultToError(CUresult(r))
}

// CuStreamWaitEvent makes a stream wait on an event without host CPU synchronization.
func CuStreamWaitEvent(hStream CUstream, hEvent CUevent, flags uint32) error {
	if err := procStreamWaitEvent.Find(); err != nil {
		return fmt.Errorf("%w: cuStreamWaitEvent: %v", ErrProcNotFound, err)
	}
	r, _, _ := procStreamWaitEvent.Call(
		uintptr(hStream),
		uintptr(hEvent),
		uintptr(flags),
	)
	return ResultToError(CUresult(r))
}

