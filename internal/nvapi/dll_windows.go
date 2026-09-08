//go:build windows

package nvapi

import (
	"errors"
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	// ErrDriverNotFound indicates nvcuda.dll is missing on the system.
	ErrDriverNotFound = errors.New("cugo: nvcuda.dll not found: NVIDIA driver not installed or incompatible")
	// ErrProcNotFound indicates a required function entry point was not exported by nvcuda.dll.
	ErrProcNotFound = errors.New("cugo: required CUDA driver symbol not found in nvcuda.dll")
)

var (
	nvcuda = windows.NewLazySystemDLL("nvcuda.dll")

	// Initialization & Device Management
	procInit               = nvcuda.NewProc("cuInit")
	procDriverGetVersion   = nvcuda.NewProc("cuDriverGetVersion")
	procDeviceGetCount     = nvcuda.NewProc("cuDeviceGetCount")
	procDeviceGet          = nvcuda.NewProc("cuDeviceGet")
	procDeviceGetName      = nvcuda.NewProc("cuDeviceGetName")
	procDeviceTotalMem     = nvcuda.NewProc("cuDeviceTotalMem_v2")
	procDeviceGetAttribute = nvcuda.NewProc("cuDeviceGetAttribute")

	// Context Management
	procCtxCreate     = nvcuda.NewProc("cuCtxCreate_v2")
	procCtxDestroy    = nvcuda.NewProc("cuCtxDestroy_v2")
	procCtxSetCurrent = nvcuda.NewProc("cuCtxSetCurrent")
	procCtxGetCurrent = nvcuda.NewProc("cuCtxGetCurrent")

	// Memory Management
	procMemAlloc        = nvcuda.NewProc("cuMemAlloc_v2")
	procMemFree         = nvcuda.NewProc("cuMemFree_v2")
	procMemcpyHtoD      = nvcuda.NewProc("cuMemcpyHtoD_v2")
	procMemcpyDtoH      = nvcuda.NewProc("cuMemcpyDtoH_v2")
	procMemcpyHtoDAsync         = nvcuda.NewProc("cuMemcpyHtoDAsync_v2")
	procMemcpyDtoHAsync         = nvcuda.NewProc("cuMemcpyDtoHAsync_v2")
	procMemAllocHost            = nvcuda.NewProc("cuMemAllocHost_v2")
	procMemFreeHost             = nvcuda.NewProc("cuMemFreeHost")
	procMemHostGetDevicePointer = nvcuda.NewProc("cuMemHostGetDevicePointer_v2")
	procMemAllocManaged         = nvcuda.NewProc("cuMemAllocManaged")
	procMemPrefetchAsync        = nvcuda.NewProc("cuMemPrefetchAsync")
	procMemAdvise               = nvcuda.NewProc("cuMemAdvise")

	// Peer-to-Peer Access
	procDeviceCanAccessPeer   = nvcuda.NewProc("cuDeviceCanAccessPeer")
	procCtxEnablePeerAccess   = nvcuda.NewProc("cuCtxEnablePeerAccess")
	procCtxDisablePeerAccess  = nvcuda.NewProc("cuCtxDisablePeerAccess")
	procMemcpyPeer            = nvcuda.NewProc("cuMemcpyPeer")
	procMemcpyPeerAsync       = nvcuda.NewProc("cuMemcpyPeerAsync")
	procDeviceGetP2PAttribute = nvcuda.NewProc("cuDeviceGetP2PAttribute")

	// Module & Kernel Launch
	procModuleLoadData   = nvcuda.NewProc("cuModuleLoadData")
	procModuleUnload     = nvcuda.NewProc("cuModuleUnload")
	procModuleGetFunction = nvcuda.NewProc("cuModuleGetFunction")
	procLaunchKernel     = nvcuda.NewProc("cuLaunchKernel")

	// Stream Management
	procStreamCreate      = nvcuda.NewProc("cuStreamCreate")
	procStreamDestroy     = nvcuda.NewProc("cuStreamDestroy_v2")
	procStreamSynchronize = nvcuda.NewProc("cuStreamSynchronize")

	// Event Management
	procEventCreate      = nvcuda.NewProc("cuEventCreate")
	procEventDestroy     = nvcuda.NewProc("cuEventDestroy_v2")
	procEventRecord      = nvcuda.NewProc("cuEventRecord")
	procEventSynchronize = nvcuda.NewProc("cuEventSynchronize")
	procEventElapsedTime = nvcuda.NewProc("cuEventElapsedTime")
)

// CheckDriver verifies whether nvcuda.dll can be dynamically loaded.
func CheckDriver() error {
	if err := nvcuda.Load(); err != nil {
		return fmt.Errorf("%w: %v", ErrDriverNotFound, err)
	}
	return nil
}

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
