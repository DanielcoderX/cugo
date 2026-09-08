//go:build windows

package driver_test

import (
	"errors"
	"math"
	"runtime"
	"testing"
	"unsafe"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/internal/nvapi"
	"github.com/cugo/cugo/kernels/vecadd"
)

func TestHostMemAllocFree(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := nvapi.CheckDriver(); err != nil {
		t.Skipf("skipped: no CUDA driver: %v", err)
	}
	if err := driver.Init(); err != nil {
		t.Skipf("skipped: driver.Init failed: %v", err)
	}

	devs, err := driver.Devices()
	if err != nil || len(devs) == 0 {
		t.Skip("skipped: no CUDA devices")
	}

	ctx, err := devs[0].CreateContext()
	if err != nil {
		t.Fatalf("dev.CreateContext failed: %v", err)
	}
	defer ctx.Destroy()

	const size = 1024 * 1024
	mem, err := ctx.AllocHost(size)
	if err != nil {
		t.Fatalf("ctx.AllocHost failed: %v", err)
	}
	if mem.Pointer() == nil {
		t.Fatal("expected non-nil pointer")
	}
	if mem.Size() != size {
		t.Fatalf("expected size %d, got %d", size, mem.Size())
	}

	b := mem.Bytes()
	if len(b) != size {
		t.Fatalf("expected slice len %d, got %d", size, len(b))
	}

	// Write pattern
	for i := range b {
		b[i] = byte(i & 0xFF)
	}

	if err := mem.Free(); err != nil {
		t.Fatalf("mem.Free failed: %v", err)
	}

	// Double-free safety
	if err := mem.Free(); !errors.Is(err, driver.ErrHostMemFreed) {
		t.Fatalf("expected ErrHostMemFreed, got: %v", err)
	}
}

func TestHostMemZeroCopyVecAdd(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := nvapi.CheckDriver(); err != nil {
		t.Skipf("skipped: no CUDA driver: %v", err)
	}
	if err := driver.Init(); err != nil {
		t.Skipf("skipped: driver.Init failed: %v", err)
	}

	devs, err := driver.Devices()
	if err != nil || len(devs) == 0 {
		t.Skip("skipped: no CUDA devices")
	}

	ctx, err := devs[0].CreateContext()
	if err != nil {
		t.Fatalf("dev.CreateContext failed: %v", err)
	}
	defer ctx.Destroy()

	// Load module
	mod, err := ctx.LoadModuleData(vecadd.PTX)
	if err != nil {
		t.Fatalf("ctx.LoadModuleData failed: %v", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("vecAdd")
	if err != nil {
		t.Fatalf("mod.Function failed: %v", err)
	}

	const n = 10000
	const byteSize = n * 4

	hMemA, err := ctx.AllocHost(byteSize); if err != nil { t.Fatal(err) }
	defer hMemA.Free()
	hMemB, err := ctx.AllocHost(byteSize); if err != nil { t.Fatal(err) }
	defer hMemB.Free()
	hMemC, err := ctx.AllocHost(byteSize); if err != nil { t.Fatal(err) }
	defer hMemC.Free()

	// Fill directly in host memory
	sliceA := unsafe.Slice((*float32)(hMemA.Pointer()), n)
	sliceB := unsafe.Slice((*float32)(hMemB.Pointer()), n)
	sliceC := unsafe.Slice((*float32)(hMemC.Pointer()), n)

	for i := 0; i < n; i++ {
		sliceA[i] = float32(i) * 3.0
		sliceB[i] = float32(i) * 7.0
	}

	// Map to GPU address space (zero-copy)
	devA, err := hMemA.DevicePointer()
	if err != nil {
		t.Fatalf("hMemA.DevicePointer failed: %v", err)
	}
	devB, err := hMemB.DevicePointer()
	if err != nil {
		t.Fatalf("hMemB.DevicePointer failed: %v", err)
	}
	devC, err := hMemC.DevicePointer()
	if err != nil {
		t.Fatalf("hMemC.DevicePointer failed: %v", err)
	}

	stream, err := ctx.CreateStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	cfg := driver.LaunchConfig{
		GridDimX:  (n + 255) / 256,
		BlockDimX: 256,
		Stream:    stream,
	}

	// Launch directly on mapped host pointers — no CopyHtoD or CopyDtoH needed!
	if err := fn.Launch(cfg, devA, devB, devC, int32(n)); err != nil {
		t.Fatalf("fn.Launch failed: %v", err)
	}

	// Synchronize stream
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream.Synchronize failed: %v", err)
	}

	// Verify directly in host memory!
	for i := 0; i < n; i++ {
		expected := sliceA[i] + sliceB[i]
		if math.Abs(float64(sliceC[i]-expected)) > 1e-4 {
			t.Fatalf("zero-copy mismatch at index %d: got %f, expected %f", i, sliceC[i], expected)
		}
	}
	t.Logf("Successfully verified %d elements via Zero-Copy Pinned Host Memory!", n)
}
