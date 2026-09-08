//go:build windows

package driver

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/cugo/cugo/internal/nvapi"
)

var (
	ErrHostMemFreed = errors.New("cugo: pinned host memory has already been freed")
)

// HostMem represents a page-locked (pinned) block of host memory.
// Page-locked host memory enables optimal DMA transfer rates with the GPU
// and can be mapped directly into GPU address space for zero-copy access.
type HostMem struct {
	ptr   unsafe.Pointer
	size  uint64
	ctx   *Context
	mu    sync.Mutex
	freed bool
}

// AllocHost allocates size bytes of page-locked host memory accessible by the GPU.
func (c *Context) AllocHost(size uint64) (*HostMem, error) {
	if size == 0 {
		return nil, ErrZeroSizeAlloc
	}
	if err := c.EnsureCurrent(); err != nil {
		return nil, err
	}

	var p unsafe.Pointer
	if err := nvapi.CuMemAllocHost(&p, size); err != nil {
		return nil, fmt.Errorf("cugo: cuMemAllocHost(%d bytes): %w", size, err)
	}

	return &HostMem{
		ptr:  p,
		size: size,
		ctx:  c,
	}, nil
}

// Free releases the page-locked host memory.
func (h *HostMem) Free() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.freed {
		return ErrHostMemFreed
	}
	h.freed = true

	if err := nvapi.CuMemFreeHost(h.ptr); err != nil {
		return fmt.Errorf("cugo: cuMemFreeHost: %w", err)
	}
	h.ptr = nil
	return nil
}

// Pointer returns the raw unsafe.Pointer to the host memory.
func (h *HostMem) Pointer() unsafe.Pointer {
	return h.ptr
}

// Size returns the allocated size in bytes.
func (h *HostMem) Size() uint64 {
	return h.size
}

// Bytes returns a byte slice backed directly by this page-locked memory buffer.
func (h *HostMem) Bytes() []byte {
	if h.ptr == nil || h.size == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(h.ptr), h.size)
}

// DevicePointer returns a device pointer address for zero-copy access by GPU kernels.
func (h *HostMem) DevicePointer(flags ...uint32) (DevicePtr, error) {
	h.mu.Lock()
	freed := h.freed
	p := h.ptr
	h.mu.Unlock()

	if freed || p == nil {
		return 0, ErrHostMemFreed
	}

	if err := h.ctx.EnsureCurrent(); err != nil {
		return 0, err
	}

	var f uint32
	if len(flags) > 0 {
		f = flags[0]
	}

	var dptr nvapi.CUdeviceptr
	if err := nvapi.CuMemHostGetDevicePointer(&dptr, p, f); err != nil {
		return 0, fmt.Errorf("cugo: cuMemHostGetDevicePointer: %w", err)
	}
	return DevicePtr(dptr), nil
}
