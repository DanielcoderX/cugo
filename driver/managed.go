//go:build windows || linux

package driver

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/cugo/cugo/internal/nvapi"
)

var (
	ErrManagedMemFreed = errors.New("cugo: managed memory has already been freed")
)

// ManagedMem represents a unified virtual address block managed by CUDA.
// The same address is accessible from both host CPU and GPU device code.
// The CUDA driver automatically migrates memory pages on demand.
type ManagedMem struct {
	dptr    DevicePtr
	hostPtr unsafe.Pointer
	size    uint64
	ctx     *Context
	mu      sync.Mutex
	freed   bool
}

// AllocManaged allocates size bytes of Unified Memory.
// Optional flags can be provided (default: CU_MEM_ATTACH_GLOBAL).
func (c *Context) AllocManaged(size uint64, flags ...uint32) (*ManagedMem, error) {
	if size == 0 {
		return nil, ErrZeroSizeAlloc
	}
	if err := c.EnsureCurrent(); err != nil {
		return nil, err
	}

	var f uint32 = nvapi.CU_MEM_ATTACH_GLOBAL
	if len(flags) > 0 {
		f = flags[0]
	}

	var rawPtr unsafe.Pointer
	if err := nvapi.CuMemAllocManaged((*nvapi.CUdeviceptr)(unsafe.Pointer(&rawPtr)), size, f); err != nil {
		return nil, fmt.Errorf("cugo: cuMemAllocManaged(%d bytes): %w", size, err)
	}

	return &ManagedMem{
		dptr:    DevicePtr(uintptr(rawPtr)),
		hostPtr: rawPtr,
		size:    size,
		ctx:     c,
	}, nil
}

// Free frees the Unified Memory buffer.
func (m *ManagedMem) Free() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.freed {
		return ErrManagedMemFreed
	}
	m.freed = true

	if err := m.ctx.EnsureCurrent(); err != nil {
		return err
	}
	if err := nvapi.CuMemFree(nvapi.CUdeviceptr(m.dptr)); err != nil {
		return fmt.Errorf("cugo: cuMemFree (managed): %w", err)
	}
	m.dptr = 0
	m.hostPtr = nil
	return nil
}

// DevicePtr returns the GPU device address for passing to kernel launches.
func (m *ManagedMem) DevicePtr() DevicePtr {
	return m.dptr
}

// Pointer returns an unsafe.Pointer to the memory for direct CPU access.
func (m *ManagedMem) Pointer() unsafe.Pointer {
	return m.hostPtr
}

// Bytes returns a byte slice directly backed by the unified memory block.
func (m *ManagedMem) Bytes() []byte {
	if m.hostPtr == nil || m.size == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(m.hostPtr), m.size)
}

// Size returns the allocated size in bytes.
func (m *ManagedMem) Size() uint64 {
	return m.size
}

// PrefetchToDevice asynchronously prefetches the unified memory range to the specified GPU device.
func (m *ManagedMem) PrefetchToDevice(dev Device, stream *Stream) error {
	m.mu.Lock()
	freed := m.freed
	dptr := m.dptr
	m.mu.Unlock()
	if freed || dptr == 0 {
		return ErrManagedMemFreed
	}

	if err := m.ctx.EnsureCurrent(); err != nil {
		return err
	}

	var hStream nvapi.CUstream
	if stream != nil {
		hStream = stream.handle
	}

	if err := nvapi.CuMemPrefetchAsync(nvapi.CUdeviceptr(dptr), m.size, dev.handle, hStream); err != nil {
		return fmt.Errorf("cugo: cuMemPrefetchAsync: %w", err)
	}
	return nil
}

// PrefetchToCPU asynchronously prefetches the unified memory range back to host CPU memory.
func (m *ManagedMem) PrefetchToCPU(stream *Stream) error {
	m.mu.Lock()
	freed := m.freed
	dptr := m.dptr
	m.mu.Unlock()
	if freed || dptr == 0 {
		return ErrManagedMemFreed
	}

	if err := m.ctx.EnsureCurrent(); err != nil {
		return err
	}

	var hStream nvapi.CUstream
	if stream != nil {
		hStream = stream.handle
	}

	if err := nvapi.CuMemPrefetchAsync(nvapi.CUdeviceptr(dptr), m.size, nvapi.CU_DEVICE_CPU, hStream); err != nil {
		return fmt.Errorf("cugo: cuMemPrefetchAsync(CPU): %w", err)
	}
	return nil
}

// Advise sets memory advising hints for the Unified Memory subsystem.
func (m *ManagedMem) Advise(advice nvapi.CUmem_advise, dev Device) error {
	m.mu.Lock()
	freed := m.freed
	dptr := m.dptr
	m.mu.Unlock()
	if freed || dptr == 0 {
		return ErrManagedMemFreed
	}

	if err := m.ctx.EnsureCurrent(); err != nil {
		return err
	}

	if err := nvapi.CuMemAdvise(nvapi.CUdeviceptr(dptr), m.size, advice, dev.handle); err != nil {
		return fmt.Errorf("cugo: cuMemAdvise: %w", err)
	}
	return nil
}
