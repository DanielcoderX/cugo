package tensor

import (
	"errors"
	"fmt"
	"sync/atomic"

	"github.com/cugo/cugo/driver"
)

var (
	ErrStorageFreed = errors.New("cugo: tensor storage has already been released")
)

// Storage manages the underlying contiguous device memory allocation for one or more Tensors.
// Multiple tensor views can safely share a single Storage via reference counting.
type Storage struct {
	ptr       driver.DevicePtr
	sizeBytes uint64
	ctx       *driver.Context
	alloc     *driver.CachingAllocator
	refCount  int32
}

// NewStorage allocates a new device memory storage block.
// If an allocator is provided, it allocates through the caching pool.
func NewStorage(ctx *driver.Context, sizeBytes uint64, alloc ...*driver.CachingAllocator) (*Storage, error) {
	if sizeBytes == 0 {
		return nil, driver.ErrZeroSizeAlloc
	}

	var ca *driver.CachingAllocator
	var dptr driver.DevicePtr
	var err error

	if len(alloc) > 0 && alloc[0] != nil {
		ca = alloc[0]
		dptr, err = ca.Alloc(sizeBytes)
	} else {
		dptr, err = ctx.Alloc(sizeBytes)
	}

	if err != nil {
		return nil, fmt.Errorf("cugo: Storage alloc (%d bytes): %w", sizeBytes, err)
	}

	s := &Storage{
		ptr:       dptr,
		sizeBytes: sizeBytes,
		ctx:       ctx,
		alloc:     ca,
		refCount:  1,
	}
	return s, nil
}

// DevicePtr returns the base GPU memory address.
func (s *Storage) DevicePtr() driver.DevicePtr {
	return s.ptr
}

// Size returns the total storage capacity in bytes.
func (s *Storage) Size() uint64 {
	return s.sizeBytes
}

// Retain increments the storage reference count.
func (s *Storage) Retain() {
	atomic.AddInt32(&s.refCount, 1)
}

// Release decrements the reference count and frees device memory when reaching zero.
func (s *Storage) Release() error {
	if atomic.LoadInt32(&s.refCount) <= 0 {
		return ErrStorageFreed
	}

	if atomic.AddInt32(&s.refCount, -1) == 0 {
		p := s.ptr
		s.ptr = 0
		if s.alloc != nil {
			return s.alloc.Free(p)
		}
		return s.ctx.Free(p)
	}
	return nil
}
