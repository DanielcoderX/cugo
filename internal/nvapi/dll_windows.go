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

	procInit               = nvcuda.NewProc("cuInit")
	procDriverGetVersion   = nvcuda.NewProc("cuDriverGetVersion")
	procDeviceGetCount     = nvcuda.NewProc("cuDeviceGetCount")
	procDeviceGet          = nvcuda.NewProc("cuDeviceGet")
	procDeviceGetName      = nvcuda.NewProc("cuDeviceGetName")
	procDeviceTotalMem     = nvcuda.NewProc("cuDeviceTotalMem_v2")
	procDeviceGetAttribute = nvcuda.NewProc("cuDeviceGetAttribute")
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

// CuDriverGetVersion returns the CUDA driver version (e.g., 12080 for 12.8, 13000 for 13.0).
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
