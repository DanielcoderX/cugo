//go:build windows || linux

package driver_test

import (
	"errors"
	"math"
	"runtime"
	"testing"
	"unsafe"

	"github.com/DanielcoderX/cugo/driver"
	"github.com/DanielcoderX/cugo/internal/nvapi"
	"github.com/DanielcoderX/cugo/kernels/vecadd"
)

func TestUnifiedMemoryAllocFree(t *testing.T) {
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
	mem, err := ctx.AllocManaged(size)
	if err != nil {
		t.Fatalf("ctx.AllocManaged failed: %v", err)
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

	// Write directly from Go CPU code
	for i := range b {
		b[i] = byte(i & 0xFF)
	}

	if err := mem.Free(); err != nil {
		t.Fatalf("mem.Free failed: %v", err)
	}

	// Double free safety
	if err := mem.Free(); !errors.Is(err, driver.ErrManagedMemFreed) {
		t.Fatalf("expected ErrManagedMemFreed, got: %v", err)
	}
}

func TestUnifiedMemoryVecAdd(t *testing.T) {
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
	dev := devs[0]

	ctx, err := dev.CreateContext()
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

	const n = 50000
	const byteSize = n * 4

	// Allocate Unified Memory for A, B, and C
	mA, err := ctx.AllocManaged(byteSize); if err != nil { t.Fatal(err) }
	defer mA.Free()
	mB, err := ctx.AllocManaged(byteSize); if err != nil { t.Fatal(err) }
	defer mB.Free()
	mC, err := ctx.AllocManaged(byteSize); if err != nil { t.Fatal(err) }
	defer mC.Free()

	// Populate inputs directly via CPU pointers
	sliceA := unsafe.Slice((*float32)(mA.Pointer()), n)
	sliceB := unsafe.Slice((*float32)(mB.Pointer()), n)
	sliceC := unsafe.Slice((*float32)(mC.Pointer()), n)

	for i := 0; i < n; i++ {
		sliceA[i] = float32(i) * 2.0
		sliceB[i] = float32(i) * 3.0
	}

	stream, err := ctx.CreateStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	// Prefetch to GPU to minimize on-demand page fault stalls
	_ = mA.PrefetchToDevice(dev, stream)
	_ = mB.PrefetchToDevice(dev, stream)

	cfg := driver.LaunchConfig{
		GridDimX:  (n + 255) / 256,
		BlockDimX: 256,
		Stream:    stream,
	}

	// Launch kernel directly on Unified Memory objects — no HtoD or DtoH memcpy calls!
	if err := fn.Launch(cfg, mA, mB, mC, int32(n)); err != nil {
		t.Fatalf("fn.Launch on Unified Memory failed: %v", err)
	}

	// Prefetch result back to CPU
	_ = mC.PrefetchToCPU(stream)

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream.Synchronize failed: %v", err)
	}

	// Verify output directly on host CPU!
	for i := 0; i < n; i++ {
		expected := sliceA[i] + sliceB[i]
		if math.Abs(float64(sliceC[i]-expected)) > 1e-4 {
			t.Fatalf("unified memory mismatch at index %d: got %f, expected %f", i, sliceC[i], expected)
		}
	}
	t.Logf("Successfully verified %d elements computed via Unified Memory!", n)
}
