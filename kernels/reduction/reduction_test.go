//go:build windows || linux

package reduction_test

import (
	"math"
	"runtime"
	"testing"
	"unsafe"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/internal/nvapi"
	"github.com/cugo/cugo/kernels/reduction"
)

func TestParallelReductionKernel(t *testing.T) {
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
		t.Fatalf("CreateContext failed: %v", err)
	}
	defer ctx.Destroy()

	mod, err := ctx.LoadModuleData(reduction.PTX)
	if err != nil {
		t.Fatalf("LoadModuleData failed: %v", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("reduce_sum")
	if err != nil {
		t.Fatalf("Function reduce_sum failed: %v", err)
	}

	const n = 1000000
	const byteSize = n * 4

	hData := make([]float32, n)
	var expectedSum float64
	for i := range hData {
		hData[i] = 1.0 // Sum should be exactly 1,000,000
		expectedSum += float64(hData[i])
	}

	dIn, err := ctx.Alloc(byteSize); if err != nil { t.Fatal(err) }
	defer ctx.Free(dIn)

	dOut, err := ctx.Alloc(4); if err != nil { t.Fatal(err) }
	defer ctx.Free(dOut)

	// Zero out output
	zero := float32(0.0)
	if err := ctx.CopyHtoD(dOut, unsafe.Slice((*byte)(unsafe.Pointer(&zero)), 4)); err != nil {
		t.Fatal(err)
	}

	rawIn := unsafe.Slice((*byte)(unsafe.Pointer(&hData[0])), byteSize)
	if err := ctx.CopyHtoD(dIn, rawIn); err != nil {
		t.Fatal(err)
	}

	cfg := driver.LaunchConfig{
		GridDimX:  64,
		BlockDimX: 256,
	}

	if err := fn.Launch(cfg, dIn, dOut, int32(n)); err != nil {
		t.Fatalf("fn.Launch failed: %v", err)
	}

	if err := ctx.Synchronize(); err != nil {
		t.Fatalf("ctx.Synchronize failed: %v", err)
	}

	var gpuSum float32
	if err := ctx.CopyDtoH(unsafe.Slice((*byte)(unsafe.Pointer(&gpuSum)), 4), dOut); err != nil {
		t.Fatal(err)
	}

	if math.Abs(float64(gpuSum)-expectedSum) > 1e-2 {
		t.Fatalf("reduction mismatch: got %f, expected %f", gpuSum, expectedSum)
	}

	t.Logf("Successfully reduced %d elements: sum = %.2f (expected %.2f)", n, gpuSum, expectedSum)
}
