//go:build windows || linux

package driver

import (
	"errors"
	"fmt"
	"sync"
)

var (
	ErrAllocatorExhausted = errors.New("cugo: caching allocator out of memory")
)

// block represents an allocated chunk of device memory in the pool.
type block struct {
	ptr   DevicePtr
	size  uint64
	inUse bool
}

// AllocatorStats provides runtime memory tracking statistics.
type AllocatorStats struct {
	AllocatedBytes uint64 // Active memory currently used by caller
	ReservedBytes  uint64 // Total memory allocated from CUDA driver
	ActiveBlocks   int    // Number of active blocks
	CachedBlocks   int    // Number of idle blocks available for reuse
}

// CachingAllocator implements a block-caching device memory allocator
// inspired by PyTorch's c10::cuda::CUDACachingAllocator.
// It drastically reduces the frequency of synchronous cuMemAlloc/cuMemFree calls.
type CachingAllocator struct {
	ctx      *Context
	mu       sync.Mutex
	blocks   map[DevicePtr]*block
	freeList []*block
	stats    AllocatorStats
}

// NewCachingAllocator initializes a caching allocator for the specified CUDA context.
func NewCachingAllocator(ctx *Context) *CachingAllocator {
	return &CachingAllocator{
		ctx:      ctx,
		blocks:   make(map[DevicePtr]*block),
		freeList: make([]*block, 0, 64),
	}
}

// roundUpToAlignment rounds bytes up to a 512-byte hardware cache line boundary.
func roundUpToAlignment(bytes uint64) uint64 {
	const align = 512
	return (bytes + align - 1) &^ (align - 1)
}

// Alloc requests a device memory block of at least size bytes.
// It first attempts to find a cached reusable block before allocating from the driver.
func (ca *CachingAllocator) Alloc(size uint64) (DevicePtr, error) {
	if size == 0 {
		return 0, ErrZeroSizeAlloc
	}
	size = roundUpToAlignment(size)

	ca.mu.Lock()
	defer ca.mu.Unlock()

	// Best-fit search on freeList
	bestIdx := -1
	var bestSize uint64 = ^uint64(0)

	for i, b := range ca.freeList {
		if b.size >= size && b.size < bestSize {
			bestIdx = i
			bestSize = b.size
			// Perfect fit or small split threshold
			if b.size == size {
				break
			}
		}
	}

	if bestIdx != -1 {
		// Reuse cached block
		b := ca.freeList[bestIdx]
		lastIdx := len(ca.freeList) - 1
		ca.freeList[bestIdx] = ca.freeList[lastIdx]
		ca.freeList = ca.freeList[:lastIdx]

		b.inUse = true
		ca.stats.AllocatedBytes += b.size
		ca.stats.ActiveBlocks++
		ca.stats.CachedBlocks--
		return b.ptr, nil
	}

	// No reusable block found: allocate new block from CUDA driver
	dptr, err := ca.ctx.Alloc(size)
	if err != nil {
		// Attempt to purge cache and retry once
		if purgeErr := ca.purgeFreeList(); purgeErr == nil {
			dptr, err = ca.ctx.Alloc(size)
		}
		if err != nil {
			return 0, fmt.Errorf("cugo: CachingAllocator failed to allocate %d bytes: %w", size, err)
		}
	}

	b := &block{
		ptr:   dptr,
		size:  size,
		inUse: true,
	}
	ca.blocks[dptr] = b
	ca.stats.AllocatedBytes += size
	ca.stats.ReservedBytes += size
	ca.stats.ActiveBlocks++

	return dptr, nil
}

// Free returns an allocated device pointer to the free pool for immediate reuse.
func (ca *CachingAllocator) Free(ptr DevicePtr) error {
	if ptr == 0 {
		return ErrNullPointer
	}

	ca.mu.Lock()
	defer ca.mu.Unlock()

	b, ok := ca.blocks[ptr]
	if !ok || !b.inUse {
		return errors.New("cugo: pointer not allocated or already returned to allocator")
	}

	b.inUse = false
	ca.freeList = append(ca.freeList, b)
	ca.stats.AllocatedBytes -= b.size
	ca.stats.ActiveBlocks--
	ca.stats.CachedBlocks++

	return nil
}

// purgeFreeList frees all idle cached blocks back to the CUDA driver.
// Caller must hold ca.mu.
func (ca *CachingAllocator) purgeFreeList() error {
	var firstErr error
	for _, b := range ca.freeList {
		delete(ca.blocks, b.ptr)
		ca.stats.ReservedBytes -= b.size
		ca.stats.CachedBlocks--
		if err := ca.ctx.Free(b.ptr); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	ca.freeList = ca.freeList[:0]
	return firstErr
}

// EmptyCache explicitly flushes all idle cached blocks back to the GPU driver.
func (ca *CachingAllocator) EmptyCache() error {
	ca.mu.Lock()
	defer ca.mu.Unlock()
	return ca.purgeFreeList()
}

// Stats returns a copy of current allocator statistics.
func (ca *CachingAllocator) Stats() AllocatorStats {
	ca.mu.Lock()
	defer ca.mu.Unlock()
	return ca.stats
}
