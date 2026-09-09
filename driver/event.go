//go:build windows || linux

package driver

import (
	"errors"
	"fmt"
	"sync"

	"github.com/DanielcoderX/cugo/internal/nvapi"
)

var (
	ErrEventDestroyed = errors.New("cugo: event has already been destroyed")
)

// Event wraps a CUDA event handle (CUevent).
type Event struct {
	handle nvapi.CUevent
	ctx    *Context
	mu     sync.Mutex
	closed bool
}

// CreateEvent creates an event within the context.
func (c *Context) CreateEvent(flags ...uint32) (*Event, error) {
	var f uint32
	if len(flags) > 0 {
		f = flags[0]
	}
	if err := c.EnsureCurrent(); err != nil {
		return nil, err
	}

	var rawEvent nvapi.CUevent
	if err := nvapi.CuEventCreate(&rawEvent, f); err != nil {
		return nil, fmt.Errorf("cugo: cuEventCreate: %w", err)
	}

	return &Event{
		handle: rawEvent,
		ctx:    c,
	}, nil
}

// Record captures the contents of the stream at the time of the call.
func (e *Event) Record(stream *Stream) error {
	if err := e.ctx.EnsureCurrent(); err != nil {
		return err
	}
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return ErrEventDestroyed
	}

	var hStream nvapi.CUstream
	if stream != nil {
		hStream = stream.handle
	}

	if err := nvapi.CuEventRecord(e.handle, hStream); err != nil {
		return fmt.Errorf("cugo: cuEventRecord: %w", err)
	}
	return nil
}

// Synchronize blocks the CPU thread until the event has occurred.
func (e *Event) Synchronize() error {
	if err := e.ctx.EnsureCurrent(); err != nil {
		return err
	}
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return ErrEventDestroyed
	}

	if err := nvapi.CuEventSynchronize(e.handle); err != nil {
		return fmt.Errorf("cugo: cuEventSynchronize: %w", err)
	}
	return nil
}

// Destroy destroys the event.
func (e *Event) Destroy() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return ErrEventDestroyed
	}
	e.closed = true

	if err := nvapi.CuEventDestroy(e.handle); err != nil {
		return fmt.Errorf("cugo: cuEventDestroy: %w", err)
	}
	e.handle = 0
	return nil
}

// ElapsedTime computes the elapsed time between start and end events in milliseconds.
func ElapsedTime(start, end *Event) (float32, error) {
	if start == nil || end == nil {
		return 0, errors.New("cugo: start and end events must not be nil")
	}

	start.mu.Lock()
	sClosed := start.closed
	hStart := start.handle
	start.mu.Unlock()

	end.mu.Lock()
	eClosed := end.closed
	hEnd := end.handle
	end.mu.Unlock()

	if sClosed || eClosed {
		return 0, ErrEventDestroyed
	}

	var ms float32
	if err := nvapi.CuEventElapsedTime(&ms, hStart, hEnd); err != nil {
		return 0, fmt.Errorf("cugo: cuEventElapsedTime: %w", err)
	}
	return ms, nil
}
