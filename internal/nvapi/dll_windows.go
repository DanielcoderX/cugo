//go:build windows

package nvapi

import (
	"errors"
	"fmt"

	"golang.org/x/sys/windows"
)

var (
	// ErrDriverNotFound indicates nvcuda.dll is missing on the system.
	ErrDriverNotFound = errors.New("cugo: nvcuda.dll not found: NVIDIA driver not installed or incompatible")
	// ErrProcNotFound indicates a required function entry point was not exported by nvcuda.dll.
	ErrProcNotFound = errors.New("cugo: required CUDA driver symbol not found in nvcuda.dll")
)

var nvcuda = windows.NewLazySystemDLL("nvcuda.dll")

// CheckDriver verifies whether nvcuda.dll can be dynamically loaded.
func CheckDriver() error {
	if err := nvcuda.Load(); err != nil {
		return fmt.Errorf("%w: %v", ErrDriverNotFound, err)
	}
	return nil
}

func newDriverProc(name string) procInvoker {
	return nvcuda.NewProc(name)
}
