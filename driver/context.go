//go:build windows

package driver

import (
	"errors"
	"fmt"
	"sync"

	"github.com/cugo/cugo/internal/nvapi"
)

var (
	ErrContextDestroyed = errors.New("cugo: CUDA context has already been destroyed")
	ErrInvalidPointer   = errors.New("cugo: invalid or null device pointer")
)

// Context wraps a CUDA driver context (CUcontext).
type Context struct {
	handle nvapi.CUcontext
	dev    Device
	mu     sync.Mutex
	closed bool
}

// CreateContext creates a new CUDA context on the given device and makes it current.
func (d Device) CreateContext(flags ...uint32) (*Context, error) {
	var f uint32
	if len(flags) > 0 {
		f = flags[0]
	}
	var rawCtx nvapi.CUcontext
	if err := nvapi.CuCtxCreate(&rawCtx, f, d.handle); err != nil {
		return nil, fmt.Errorf("cugo: cuCtxCreate: %w", err)
	}
	return &Context{
		handle: rawCtx,
		dev:    d,
	}, nil
}

// CurrentContext retrieves the CUDA context currently bound to the calling thread.
func CurrentContext() (*Context, error) {
	var rawCtx nvapi.CUcontext
	if err := nvapi.CuCtxGetCurrent(&rawCtx); err != nil {
		return nil, fmt.Errorf("cugo: cuCtxGetCurrent: %w", err)
	}
	if rawCtx == 0 {
		return nil, nil
	}
	return &Context{handle: rawCtx}, nil
}

// Destroy destroys the CUDA context. Calling Destroy multiple times returns an error.
func (c *Context) Destroy() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrContextDestroyed
	}
	c.closed = true
	h := c.handle
	c.handle = 0
	if err := nvapi.CuCtxDestroy(h); err != nil {
		return fmt.Errorf("cugo: cuCtxDestroy: %w", err)
	}
	_ = nvapi.CuCtxSetCurrent(0)
	return nil
}

// SetCurrent binds this CUDA context to the calling CPU thread.
func (c *Context) SetCurrent() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrContextDestroyed
	}
	if err := nvapi.CuCtxSetCurrent(c.handle); err != nil {
		return fmt.Errorf("cugo: cuCtxSetCurrent: %w", err)
	}
	return nil
}

// EnsureCurrent ensures that this CUDA context is active on the calling OS thread.
func (c *Context) EnsureCurrent() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrContextDestroyed
	}
	if err := nvapi.CuCtxSetCurrent(c.handle); err != nil {
		return fmt.Errorf("cugo: cuCtxSetCurrent: %w", err)
	}
	return nil
}

// Device returns the Device on which this Context was created.
func (c *Context) Device() Device {
	return c.dev
}

// Handle returns the underlying raw CUcontext.
func (c *Context) Handle() uintptr {
	return uintptr(c.handle)
}

// Synchronize blocks until the device has completed all preceding requested tasks in this context.
func (c *Context) Synchronize() error {
	if err := c.EnsureCurrent(); err != nil {
		return err
	}
	return nvapi.CuCtxSynchronize()
}
