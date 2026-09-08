//go:build windows

package driver

import (
	"errors"
	"unsafe"

	"github.com/cugo/cugo/internal/nvapi"
)

var (
	ErrMemPoolNil       = errors.New("cugo: nil memory pool handle")
	ErrMemPoolDestroyed = errors.New("cugo: memory pool already destroyed")
)

// AllocAsync allocates memory asynchronously on the specified stream.
// If stream is nil, the default stream is used.
func (c *Context) AllocAsync(bytesize uint64, stream *Stream) (DevicePtr, error) {
	if c == nil || c.handle == 0 {
		return 0, ErrContextDestroyed
	}
	if err := c.EnsureCurrent(); err != nil {
		return 0, err
	}
	var hStream nvapi.CUstream
	if stream != nil {
		hStream = stream.handle
	}
	var dptr nvapi.CUdeviceptr
	if err := nvapi.CuMemAllocAsync(&dptr, bytesize, hStream); err != nil {
		return 0, err
	}
	return DevicePtr(dptr), nil
}

// FreeAsync frees memory asynchronously on the specified stream.
// If stream is nil, the default stream is used.
func (c *Context) FreeAsync(ptr DevicePtr, stream *Stream) error {
	if c == nil || c.handle == 0 {
		return ErrContextDestroyed
	}
	if err := c.EnsureCurrent(); err != nil {
		return err
	}
	var hStream nvapi.CUstream
	if stream != nil {
		hStream = stream.handle
	}
	return nvapi.CuMemFreeAsync(nvapi.CUdeviceptr(ptr), hStream)
}

// MemPool wraps a CUDA stream-ordered memory pool handle.
type MemPool struct {
	handle nvapi.CUmemoryPool
}

// DefaultMemPool returns the device's default memory pool.
func (d Device) DefaultMemPool() (*MemPool, error) {
	var pool nvapi.CUmemoryPool
	if err := nvapi.CuDeviceGetDefaultMemPool(&pool, nvapi.CUdevice(d.handle)); err != nil {
		return nil, err
	}
	return &MemPool{handle: pool}, nil
}

// TrimTo releases unallocated memory held by the pool back to the OS/driver,
// keeping at least minBytesToKeep cached in the pool.
func (mp *MemPool) TrimTo(minBytesToKeep uint64) error {
	if mp == nil || mp.handle == 0 {
		return ErrMemPoolNil
	}
	return nvapi.CuMemPoolTrimTo(mp.handle, minBytesToKeep)
}

// ReservedMemCurrent returns the current amount of GPU memory reserved by the pool in bytes.
func (mp *MemPool) ReservedMemCurrent() (uint64, error) {
	if mp == nil || mp.handle == 0 {
		return 0, ErrMemPoolNil
	}
	var val uint64
	if err := nvapi.CuMemPoolGetAttribute(mp.handle, nvapi.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT, unsafe.Pointer(&val)); err != nil {
		return 0, err
	}
	return val, nil
}

// UsedMemCurrent returns the amount of memory currently allocated to active allocations in bytes.
func (mp *MemPool) UsedMemCurrent() (uint64, error) {
	if mp == nil || mp.handle == 0 {
		return 0, ErrMemPoolNil
	}
	var val uint64
	if err := nvapi.CuMemPoolGetAttribute(mp.handle, nvapi.CU_MEMPOOL_ATTR_USED_MEM_CURRENT, unsafe.Pointer(&val)); err != nil {
		return 0, err
	}
	return val, nil
}

// ReleaseThreshold returns the amount of reserved memory in bytes that the pool will attempt to retain
// when releasing memory back to the OS.
func (mp *MemPool) ReleaseThreshold() (uint64, error) {
	if mp == nil || mp.handle == 0 {
		return 0, ErrMemPoolNil
	}
	var val uint64
	if err := nvapi.CuMemPoolGetAttribute(mp.handle, nvapi.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, unsafe.Pointer(&val)); err != nil {
		return 0, err
	}
	return val, nil
}

// SetReleaseThreshold sets the release threshold in bytes.
func (mp *MemPool) SetReleaseThreshold(threshold uint64) error {
	if mp == nil || mp.handle == 0 {
		return ErrMemPoolNil
	}
	return nvapi.CuMemPoolSetAttribute(mp.handle, nvapi.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, unsafe.Pointer(&threshold))
}
