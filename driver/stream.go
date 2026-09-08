//go:build windows

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
	handle nvapi.CUstream
	ctx    *Context
	mu     sync.Mutex
	closed bool
}

// CreateStream creates an execution stream in this context.
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
