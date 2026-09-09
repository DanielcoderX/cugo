//go:build windows || linux

package driver

import (
	"errors"
	"fmt"

	"github.com/DanielcoderX/cugo/internal/nvapi"
)

var (
	ErrZeroSizeAlloc = errors.New("cugo: allocation size must be greater than zero")
	ErrNullPointer   = errors.New("cugo: cannot operate on null DevicePtr")
)

// DevicePtr represents a 64-bit device memory address (mirrors CUdeviceptr).
type DevicePtr uintptr

// Uintptr returns the numeric address of the DevicePtr.
func (p DevicePtr) Uintptr() uintptr {
	return uintptr(p)
}

// IsNil reports whether the pointer is 0 (null).
func (p DevicePtr) IsNil() bool {
	return p == 0
}

// Offset returns a new DevicePtr advanced by the specified byte offset.
func (p DevicePtr) Offset(bytes uint64) DevicePtr {
	return DevicePtr(uintptr(p) + uintptr(bytes))
}

// Add returns a new DevicePtr offset by the specified signed byte count.
func (p DevicePtr) Add(bytes int64) DevicePtr {
	return DevicePtr(uintptr(int64(p) + bytes))
}

// Alloc allocates size bytes of linear memory on the GPU within this context.
func (c *Context) Alloc(size uint64) (DevicePtr, error) {
	if size == 0 {
		return 0, ErrZeroSizeAlloc
	}
	if err := c.EnsureCurrent(); err != nil {
		return 0, err
	}

	var dptr nvapi.CUdeviceptr
	if err := nvapi.CuMemAlloc(&dptr, size); err != nil {
		return 0, fmt.Errorf("cugo: cuMemAlloc(%d bytes): %w", size, err)
	}
	return DevicePtr(dptr), nil
}

// Free frees linear memory previously allocated on the GPU.
func (c *Context) Free(ptr DevicePtr) error {
	if ptr == 0 {
		return ErrNullPointer
	}
	if err := c.EnsureCurrent(); err != nil {
		return err
	}

	if err := nvapi.CuMemFree(nvapi.CUdeviceptr(ptr)); err != nil {
		return fmt.Errorf("cugo: cuMemFree(0x%x): %w", uintptr(ptr), err)
	}
	return nil
}

// CopyHtoD synchronously copies byte slice src from host CPU memory to GPU device memory dst.
func (c *Context) CopyHtoD(dst DevicePtr, src []byte) error {
	if dst == 0 {
		return ErrNullPointer
	}
	if len(src) == 0 {
		return nil
	}
	if err := c.EnsureCurrent(); err != nil {
		return err
	}

	if err := nvapi.CuMemcpyHtoD(nvapi.CUdeviceptr(dst), src); err != nil {
		return fmt.Errorf("cugo: cuMemcpyHtoD: %w", err)
	}
	return nil
}

// CopyDtoH synchronously copies dst bytes from GPU device memory src to host CPU memory dst.
func (c *Context) CopyDtoH(dst []byte, src DevicePtr) error {
	if src == 0 {
		return ErrNullPointer
	}
	if len(dst) == 0 {
		return nil
	}
	if err := c.EnsureCurrent(); err != nil {
		return err
	}

	if err := nvapi.CuMemcpyDtoH(dst, nvapi.CUdeviceptr(src)); err != nil {
		return fmt.Errorf("cugo: cuMemcpyDtoH: %w", err)
	}
	return nil
}

// MemsetD8 sets count bytes of device memory to value synchronously.
func (c *Context) MemsetD8(dst DevicePtr, value uint8, count uint64) error {
	if dst == 0 {
		return ErrNullPointer
	}
	if count == 0 {
		return nil
	}
	if err := c.EnsureCurrent(); err != nil {
		return err
	}
	if err := nvapi.CuMemsetD8Async(nvapi.CUdeviceptr(dst), value, count, 0); err != nil {
		return fmt.Errorf("cugo: cuMemsetD8Async: %w", err)
	}
	return c.Synchronize()
}

// MemsetD32 sets count 32-bit words of device memory to value synchronously.
func (c *Context) MemsetD32(dst DevicePtr, value uint32, count uint64) error {
	if dst == 0 {
		return ErrNullPointer
	}
	if count == 0 {
		return nil
	}
	if err := c.EnsureCurrent(); err != nil {
		return err
	}
	if err := nvapi.CuMemsetD32Async(nvapi.CUdeviceptr(dst), value, count, 0); err != nil {
		return fmt.Errorf("cugo: cuMemsetD32Async: %w", err)
	}
	return c.Synchronize()
}

