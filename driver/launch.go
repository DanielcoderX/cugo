//go:build windows || linux

package driver

import (
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"unsafe"

	"github.com/cugo/cugo/internal/nvapi"
)

// LaunchConfig specifies grid dimensions, block dimensions, dynamic shared memory, and stream.
type LaunchConfig struct {
	GridDimX, GridDimY, GridDimZ    uint32
	BlockDimX, BlockDimY, BlockDimZ uint32
	SharedMemBytes                  uint32
	Stream                          *Stream // nil specifies default stream
}

// KernelArg represents an individual typed kernel argument.
type KernelArg interface {
	argPtr() unsafe.Pointer
}

type ptrArg struct{ val uintptr }

func (a *ptrArg) argPtr() unsafe.Pointer { return unsafe.Pointer(&a.val) }

// Ptr creates a KernelArg for a GPU DevicePtr (CUdeviceptr / void*).
func Ptr(p DevicePtr) KernelArg { return &ptrArg{val: uintptr(p)} }

type int32Arg struct{ val int32 }

func (a *int32Arg) argPtr() unsafe.Pointer { return unsafe.Pointer(&a.val) }

// Int32 creates a KernelArg for a 32-bit signed integer.
func Int32(v int32) KernelArg { return &int32Arg{val: v} }

type uint32Arg struct{ val uint32 }

func (a *uint32Arg) argPtr() unsafe.Pointer { return unsafe.Pointer(&a.val) }

// Uint32 creates a KernelArg for a 32-bit unsigned integer.
func Uint32(v uint32) KernelArg { return &uint32Arg{val: v} }

type int64Arg struct{ val int64 }

func (a *int64Arg) argPtr() unsafe.Pointer { return unsafe.Pointer(&a.val) }

// Int64 creates a KernelArg for a 64-bit signed integer.
func Int64(v int64) KernelArg { return &int64Arg{val: v} }

type uint64Arg struct{ val uint64 }

func (a *uint64Arg) argPtr() unsafe.Pointer { return unsafe.Pointer(&a.val) }

// Uint64 creates a KernelArg for a 64-bit unsigned integer.
func Uint64(v uint64) KernelArg { return &uint64Arg{val: v} }

type float32Arg struct{ val float32 }

func (a *float32Arg) argPtr() unsafe.Pointer { return unsafe.Pointer(&a.val) }

// Float32 creates a KernelArg for a 32-bit floating point number.
func Float32(v float32) KernelArg { return &float32Arg{val: v} }

type float64Arg struct{ val float64 }

func (a *float64Arg) argPtr() unsafe.Pointer { return unsafe.Pointer(&a.val) }

// Float64 creates a KernelArg for a 64-bit floating point number.
func Float64(v float64) KernelArg { return &float64Arg{val: v} }

type rawArg struct{ p unsafe.Pointer }

func (a *rawArg) argPtr() unsafe.Pointer { return a.p }

// Raw creates a KernelArg pointing directly to an existing memory buffer.
func Raw(p unsafe.Pointer) KernelArg { return &rawArg{p: p} }

// Bool creates a KernelArg for an 8-bit boolean.
func Bool(v bool) KernelArg {
	var b uint8
	if v {
		b = 1
	}
	return &uint8Arg{val: b}
}

type int8Arg struct{ val int8 }

func (a *int8Arg) argPtr() unsafe.Pointer { return unsafe.Pointer(&a.val) }

// Int8 creates a KernelArg for an 8-bit signed integer.
func Int8(v int8) KernelArg { return &int8Arg{val: v} }

type uint8Arg struct{ val uint8 }

func (a *uint8Arg) argPtr() unsafe.Pointer { return unsafe.Pointer(&a.val) }

// Uint8 creates a KernelArg for an 8-bit unsigned integer (byte).
func Uint8(v uint8) KernelArg { return &uint8Arg{val: v} }

type int16Arg struct{ val int16 }

func (a *int16Arg) argPtr() unsafe.Pointer { return unsafe.Pointer(&a.val) }

// Int16 creates a KernelArg for a 16-bit signed integer.
func Int16(v int16) KernelArg { return &int16Arg{val: v} }

type uint16Arg struct{ val uint16 }

func (a *uint16Arg) argPtr() unsafe.Pointer { return unsafe.Pointer(&a.val) }

// Uint16 creates a KernelArg for a 16-bit unsigned integer.
func Uint16(v uint16) KernelArg { return &uint16Arg{val: v} }

func toKernelArg(v any) (KernelArg, error) {
	if v == nil {
		return nil, errors.New("cugo: nil kernel argument is not supported")
	}

	switch val := v.(type) {
	case KernelArg:
		return val, nil
	case DevicePtr:
		return Ptr(val), nil
	case uintptr:
		return Ptr(DevicePtr(val)), nil
	case *HostMem:
		dptr, err := val.DevicePointer()
		if err != nil {
			return nil, err
		}
		return Ptr(dptr), nil
	case *ManagedMem:
		return Ptr(val.DevicePtr()), nil
	case *TextureObject:
		return Uint64(val.Handle()), nil
	case *SurfaceObject:
		return Uint64(val.Handle()), nil
	case bool:
		return Bool(val), nil
	case int8:
		return Int8(val), nil
	case uint8:
		return Uint8(val), nil
	case int16:
		return Int16(val), nil
	case uint16:
		return Uint16(val), nil
	case int32:
		return Int32(val), nil
	case int:
		return Int32(int32(val)), nil
	case uint32:
		return Uint32(val), nil
	case uint:
		return Uint32(uint32(val)), nil
	case int64:
		return Int64(val), nil
	case uint64:
		return Uint64(val), nil
	case float32:
		return Float32(val), nil
	case float64:
		return Float64(val), nil
	case unsafe.Pointer:
		return Raw(val), nil
	default:
		// Reflection fallback for structs & pointer to structs
		rv := reflect.ValueOf(v)
		if rv.Kind() == reflect.Struct {
			copyVal := reflect.New(rv.Type())
			copyVal.Elem().Set(rv)
			return Raw(copyVal.UnsafePointer()), nil
		}
		if rv.Kind() == reflect.Pointer && rv.Elem().Kind() == reflect.Struct {
			return Raw(rv.UnsafePointer()), nil
		}
		return nil, fmt.Errorf("cugo: unsupported argument type %T (use driver.Raw if custom memory)", v)
	}
}

// Launch launches the kernel on the GPU using the specified launch configuration and arguments.
func (f *Function) Launch(cfg LaunchConfig, args ...any) error {
	if f == nil || f.handle == 0 {
		return errors.New("cugo: nil or uninitialized Function")
	}

	if err := f.mod.ctx.EnsureCurrent(); err != nil {
		return err
	}
	f.mod.mu.Lock()
	unloaded := f.mod.unloaded
	f.mod.mu.Unlock()
	if unloaded {
		return ErrModuleUnloaded
	}

	gridX := cfg.GridDimX
	if gridX == 0 {
		gridX = 1
	}
	gridY := cfg.GridDimY
	if gridY == 0 {
		gridY = 1
	}
	gridZ := cfg.GridDimZ
	if gridZ == 0 {
		gridZ = 1
	}

	blockX := cfg.BlockDimX
	if blockX == 0 {
		blockX = 1
	}
	blockY := cfg.BlockDimY
	if blockY == 0 {
		blockY = 1
	}
	blockZ := cfg.BlockDimZ
	if blockZ == 0 {
		blockZ = 1
	}

	var kernelParams unsafe.Pointer
	var paramPtrs []unsafe.Pointer
	var kargs []KernelArg

	if len(args) > 0 {
		paramPtrs = make([]unsafe.Pointer, len(args))
		kargs = make([]KernelArg, len(args))

		for i, arg := range args {
			karg, err := toKernelArg(arg)
			if err != nil {
				return fmt.Errorf("cugo: kernel argument %d: %w", i, err)
			}
			kargs[i] = karg
			paramPtrs[i] = karg.argPtr()
		}
		kernelParams = unsafe.Pointer(&paramPtrs[0])
	}

	var hStream nvapi.CUstream
	if cfg.Stream != nil {
		hStream = cfg.Stream.handle
	}

	err := nvapi.CuLaunchKernel(
		f.handle,
		gridX, gridY, gridZ,
		blockX, blockY, blockZ,
		cfg.SharedMemBytes,
		hStream,
		kernelParams,
		nil,
	)

	runtime.KeepAlive(kargs)
	runtime.KeepAlive(paramPtrs)

	if err != nil {
		return fmt.Errorf("cugo: cuLaunchKernel(%s): %w", f.name, err)
	}

	return nil
}
