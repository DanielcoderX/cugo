//go:build windows

package driver

import (
	"bytes"
	"errors"
	"fmt"
	"sync"

	"github.com/cugo/cugo/internal/nvapi"
)

var (
	ErrModuleUnloaded = errors.New("cugo: module has already been unloaded")
)

// Module wraps a loaded CUDA module handle (CUmodule).
type Module struct {
	handle   nvapi.CUmodule
	ctx      *Context
	mu       sync.Mutex
	unloaded bool
}

// Function wraps a CUDA kernel entry point handle (CUfunction).
type Function struct {
	handle nvapi.CUfunction
	name   string
	mod    *Module
}

// LoadModuleData loads a module from raw PTX or cubin bytecode within the current context.
func (c *Context) LoadModuleData(ptxOrCubin []byte) (*Module, error) {
	if len(ptxOrCubin) == 0 {
		return nil, errors.New("cugo: cannot load empty module data")
	}

	if err := c.EnsureCurrent(); err != nil {
		return nil, err
	}

	// Ensure null-termination for PTX string if not already present
	data := ptxOrCubin
	if !bytes.HasSuffix(data, []byte{0}) {
		data = append(bytes.Clone(ptxOrCubin), 0)
	}

	var rawMod nvapi.CUmodule
	if err := nvapi.CuModuleLoadData(&rawMod, data); err != nil {
		return nil, fmt.Errorf("cugo: cuModuleLoadData: %w", err)
	}

	return &Module{
		handle: rawMod,
		ctx:    c,
	}, nil
}

// Function returns a handle to a kernel function defined within the module.
func (m *Module) Function(name string) (*Function, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.unloaded {
		return nil, ErrModuleUnloaded
	}

	var rawFunc nvapi.CUfunction
	if err := nvapi.CuModuleGetFunction(&rawFunc, m.handle, name); err != nil {
		return nil, fmt.Errorf("cugo: cuModuleGetFunction(%q): %w", name, err)
	}

	return &Function{
		handle: rawFunc,
		name:   name,
		mod:    m,
	}, nil
}

// Unload unloads the module from the GPU context.
func (m *Module) Unload() error {
	if err := m.ctx.EnsureCurrent(); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.unloaded {
		return ErrModuleUnloaded
	}
	m.unloaded = true

	if err := nvapi.CuModuleUnload(m.handle); err != nil {
		return fmt.Errorf("cugo: cuModuleUnload: %w", err)
	}
	m.handle = 0
	return nil
}

// Name returns the kernel function name.
func (f *Function) Name() string {
	return f.name
}

// Handle returns the underlying raw CUfunction.
func (f *Function) Handle() uintptr {
	return uintptr(f.handle)
}
