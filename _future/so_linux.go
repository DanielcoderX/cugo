//go:build ignore

package future

import (
	"errors"
	"fmt"
	"sync"

	"github.com/ebitengine/purego"
)

var (
	ErrDriverNotFound = errors.New("cugo: libcuda.so not found: NVIDIA driver not installed or incompatible")
	ErrProcNotFound   = errors.New("cugo: required CUDA driver symbol not found in libcuda.so")
)

var (
	libcuda     uintptr
	libcudaOnce sync.Once
	libcudaErr  error
)

func CheckDriver() error {
	_, err := openLibcuda()
	return err
}

func openLibcuda() (uintptr, error) {
	libcudaOnce.Do(func() {
		candidates := []string{
			"libcuda.so.1",
			"libcuda.so",
			"/usr/lib/x86_64-linux-gnu/libcuda.so.1",
			"/usr/lib64/libcuda.so.1",
			"/usr/lib/wsl/lib/libcuda.so.1",
		}
		for _, name := range candidates {
			h, err := purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
			if err == nil && h != 0 {
				libcuda = h
				return
			}
		}
		libcudaErr = ErrDriverNotFound
	})
	return libcuda, libcudaErr
}

type linuxLazyProc struct {
	name string
	addr uintptr
	err  error
	once sync.Once
}

func (p *linuxLazyProc) Find() error {
	p.once.Do(func() {
		h, err := openLibcuda()
		if err != nil {
			p.err = err
			return
		}
		sym, err := purego.Dlsym(h, p.name)
		if err != nil || sym == 0 {
			p.err = fmt.Errorf("%w: %s: %v", ErrProcNotFound, p.name, err)
			return
		}
		p.addr = sym
	})
	return p.err
}

func (p *linuxLazyProc) Call(args ...uintptr) (r1, r2 uintptr, lastErr error) {
	if err := p.Find(); err != nil {
		return 0, 0, err
	}
	r1, r2, _ = purego.SyscallN(p.addr, args...)
	return r1, r2, nil
}
