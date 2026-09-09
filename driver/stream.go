//go:build windows || linux

package driver

import (
	"errors"
	"fmt"
	"sync"

	"github.com/cugo/cugo/internal/nvapi"
)

var (
	ErrStreamDestroyed = errors.New("cugo: stream has already been destroyed")
)

// Stream wraps a CUDA asynchronous execution stream (CUstream).
type Stream struct {
	handle   nvapi.CUstream
	ctx      *Context
	mu       sync.Mutex
	closed   bool
	priority int
}

// Priority returns the scheduling priority assigned to this stream.
func (s *Stream) Priority() int {
	return s.priority
}

// StreamPriorityRange queries the numerical range of valid stream priorities on this context.
// In CUDA, lower numerical values designate higher scheduling priority.
func (c *Context) StreamPriorityRange() (least, greatest int, err error) {
	if err := c.EnsureCurrent(); err != nil {
		return 0, 0, err
	}
	var lp, gp int32
	if err := nvapi.CuCtxGetStreamPriorityRange(&lp, &gp); err != nil {
		return 0, 0, fmt.Errorf("cugo: cuCtxGetStreamPriorityRange: %w", err)
	}
	return int(lp), int(gp), nil
}

// CreateStream creates an execution stream in this context using default priority.
func (c *Context) CreateStream(flags ...uint32) (*Stream, error) {
	var f uint32
	if len(flags) > 0 {
		f = flags[0]
	}
	if err := c.EnsureCurrent(); err != nil {
		return nil, err
	}

	var rawStream nvapi.CUstream
	if err := nvapi.CuStreamCreate(&rawStream, f); err != nil {
		return nil, fmt.Errorf("cugo: cuStreamCreate: %w", err)
	}

	return &Stream{
		handle: rawStream,
		ctx:    c,
	}, nil
}

// CreateStreamWithPriority creates an execution stream in this context with the specified scheduling priority.
func (c *Context) CreateStreamWithPriority(priority int, flags ...uint32) (*Stream, error) {
	var f uint32
	if len(flags) > 0 {
		f = flags[0]
	}
	if err := c.EnsureCurrent(); err != nil {
		return nil, err
	}

	var rawStream nvapi.CUstream
	if err := nvapi.CuStreamCreateWithPriority(&rawStream, f, int32(priority)); err != nil {
		return nil, fmt.Errorf("cugo: cuStreamCreateWithPriority(%d): %w", priority, err)
	}

	return &Stream{
		handle:   rawStream,
		ctx:      c,
		priority: priority,
	}, nil
}


// Synchronize blocks until all operations queued in this stream have completed.
func (s *Stream) Synchronize() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrStreamDestroyed
	}
	if err := nvapi.CuStreamSynchronize(s.handle); err != nil {
		return fmt.Errorf("cugo: cuStreamSynchronize: %w", err)
	}
	return nil
}

// Destroy destroys the stream.
func (s *Stream) Destroy() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrStreamDestroyed
	}
	s.closed = true

	if err := nvapi.CuStreamDestroy(s.handle); err != nil {
		return fmt.Errorf("cugo: cuStreamDestroy: %w", err)
	}
	s.handle = 0
	return nil
}

// Handle returns the underlying CUstream.
func (s *Stream) Handle() uintptr {
	if s == nil {
		return 0 // default stream
	}
	return uintptr(s.handle)
}

// CopyHtoDAsync copies memory from host CPU to device asynchronously on this stream.
func (s *Stream) CopyHtoDAsync(dst DevicePtr, src []byte) error {
	if dst == 0 {
		return ErrNullPointer
	}
	if len(src) == 0 {
		return nil
	}
	if err := s.ctx.EnsureCurrent(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrStreamDestroyed
	}
	if err := nvapi.CuMemcpyHtoDAsync(nvapi.CUdeviceptr(dst), src, s.handle); err != nil {
		return fmt.Errorf("cugo: cuMemcpyHtoDAsync: %w", err)
	}
	return nil
}

// CopyDtoHAsync copies memory from device to host CPU asynchronously on this stream.
func (s *Stream) CopyDtoHAsync(dst []byte, src DevicePtr) error {
	if src == 0 {
		return ErrNullPointer
	}
	if len(dst) == 0 {
		return nil
	}
	if err := s.ctx.EnsureCurrent(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrStreamDestroyed
	}
	if err := nvapi.CuMemcpyDtoHAsync(dst, nvapi.CUdeviceptr(src), s.handle); err != nil {
		return fmt.Errorf("cugo: cuMemcpyDtoHAsync: %w", err)
	}
	return nil
}

// MemsetD8Async sets count bytes of device memory to value asynchronously on this stream.
func (s *Stream) MemsetD8Async(dst DevicePtr, value uint8, count uint64) error {
	if dst == 0 {
		return ErrNullPointer
	}
	if count == 0 {
		return nil
	}
	if err := s.ctx.EnsureCurrent(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrStreamDestroyed
	}
	if err := nvapi.CuMemsetD8Async(nvapi.CUdeviceptr(dst), value, count, s.handle); err != nil {
		return fmt.Errorf("cugo: cuMemsetD8Async: %w", err)
	}
	return nil
}

// MemsetD32Async sets count 32-bit words of device memory to value asynchronously on this stream.
func (s *Stream) MemsetD32Async(dst DevicePtr, value uint32, count uint64) error {
	if dst == 0 {
		return ErrNullPointer
	}
	if count == 0 {
		return nil
	}
	if err := s.ctx.EnsureCurrent(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrStreamDestroyed
	}
	if err := nvapi.CuMemsetD32Async(nvapi.CUdeviceptr(dst), value, count, s.handle); err != nil {
		return fmt.Errorf("cugo: cuMemsetD32Async: %w", err)
	}
	return nil
}

// WaitEvent makes this stream wait for the specified event before executing subsequent operations.
// The wait is performed entirely on the GPU without blocking the host CPU thread.
func (s *Stream) WaitEvent(event *Event) error {
	if event == nil || event.handle == 0 {
		return errors.New("cugo: nil or uninitialized Event")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrStreamDestroyed
	}
	if err := nvapi.CuStreamWaitEvent(s.handle, event.handle, 0); err != nil {
		return fmt.Errorf("cugo: cuStreamWaitEvent: %w", err)
	}
	return nil
}

