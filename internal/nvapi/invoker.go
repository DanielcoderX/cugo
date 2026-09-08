//go:build windows || linux

package nvapi

type procInvoker interface {
	Find() error
	Call(a ...uintptr) (r1, r2 uintptr, lastErr error)
}
