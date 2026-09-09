//go:build windows || linux

package driver

import (
	"errors"
	"sync"
	"unsafe"

	"github.com/cugo/cugo/internal/nvapi"
)

var (
	ErrLinkerNil       = errors.New("cugo: nil linker state")
	ErrLinkerDestroyed = errors.New("cugo: linker state already destroyed")
)

// Linker represents a pending CUDA JIT linker invocation for dynamic runtime linking of PTX and CUBIN code.
type Linker struct {
	mu        sync.Mutex
	handle    nvapi.CUlinkState
	destroyed bool
}

// CreateLinker initializes a new JIT linker invocation in the current context.
func (c *Context) CreateLinker() (*Linker, error) {
	if c == nil || c.handle == 0 {
		return nil, ErrContextDestroyed
	}
	if err := c.EnsureCurrent(); err != nil {
		return nil, err
	}
	var state nvapi.CUlinkState
	if err := nvapi.CuLinkCreate(0, nil, nil, &state); err != nil {
		return nil, err
	}
	return &Linker{handle: state}, nil
}

// AddPTX adds PTX source code bytes to the pending link.
func (l *Linker) AddPTX(ptx []byte, name string) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.destroyed || l.handle == 0 {
		return ErrLinkerDestroyed
	}
	return nvapi.CuLinkAddData(l.handle, nvapi.CU_JIT_INPUT_PTX, ptx, name, 0, nil, nil)
}

// AddCubin adds precompiled CUBIN bytecode to the pending link.
func (l *Linker) AddCubin(cubin []byte, name string) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.destroyed || l.handle == 0 {
		return ErrLinkerDestroyed
	}
	return nvapi.CuLinkAddData(l.handle, nvapi.CU_JIT_INPUT_CUBIN, cubin, name, 0, nil, nil)
}

// Complete finalizes linking and returns an owned copy of the linked CUBIN binary.
func (l *Linker) Complete() ([]byte, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.destroyed || l.handle == 0 {
		return nil, ErrLinkerDestroyed
	}
	var cubinPtr unsafe.Pointer
	var size uint64
	if err := nvapi.CuLinkComplete(l.handle, &cubinPtr, &size); err != nil {
		return nil, err
	}
	if size == 0 || cubinPtr == nil {
		return nil, errors.New("cugo: linker returned empty cubin")
	}
	srcSlice := unsafe.Slice((*byte)(cubinPtr), size)
	out := make([]byte, size)
	copy(out, srcSlice)
	return out, nil
}

// Destroy frees the JIT linker invocation state.
func (l *Linker) Destroy() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.destroyed {
		return ErrLinkerDestroyed
	}
	l.destroyed = true
	if l.handle != 0 {
		h := l.handle
		l.handle = 0
		return nvapi.CuLinkDestroy(h)
	}
	return nil
}
