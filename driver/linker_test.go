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

func TestDynamicJITLinker(t *testing.T) {
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

	// 1. Create Linker
	linker, err := ctx.CreateLinker()
	if err != nil {
		t.Fatalf("CreateLinker failed: %v", err)
	}
	defer linker.Destroy()

	// 2. Add PTX source code
	if err := linker.AddPTX(vecadd.PTX, "vecadd.ptx"); err != nil {
		t.Fatalf("linker.AddPTX failed: %v", err)
	}

	// 3. Complete JIT link to native device CUBIN
	cubin, err := linker.Complete()
	if err != nil {
		t.Fatalf("linker.Complete failed: %v", err)
	}
	if len(cubin) == 0 {
		t.Fatal("expected non-empty cubin bytecode")
	}
	t.Logf("Dynamically JIT-linked PTX into %d bytes CUBIN", len(cubin))

	// 4. Destroy linker safely
	if err := linker.Destroy(); err != nil {
		t.Fatalf("linker.Destroy failed: %v", err)
	}
	if err := linker.Destroy(); !errors.Is(err, driver.ErrLinkerDestroyed) {
		t.Fatalf("expected ErrLinkerDestroyed on double destroy, got: %v", err)
	}

	// 5. Load the JIT-linked CUBIN directly as a module
	mod, err := ctx.LoadModuleData(cubin)
	if err != nil {
		t.Fatalf("LoadModuleData with JIT CUBIN failed: %v", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("vecAdd")
	if err != nil {
		t.Fatalf("Function vecAdd from JIT CUBIN failed: %v", err)
	}

	// 6. Launch kernel from JIT-linked module to verify functionality
	const n = 1000
	const byteSize = n * 4

	mA, err := ctx.AllocManaged(byteSize); if err != nil { t.Fatal(err) }
	defer mA.Free()
	mB, err := ctx.AllocManaged(byteSize); if err != nil { t.Fatal(err) }
	defer mB.Free()
	mC, err := ctx.AllocManaged(byteSize); if err != nil { t.Fatal(err) }
	defer mC.Free()

	sliceA := unsafe.Slice((*float32)(mA.Pointer()), n)
	sliceB := unsafe.Slice((*float32)(mB.Pointer()), n)
	sliceC := unsafe.Slice((*float32)(mC.Pointer()), n)

	for i := 0; i < n; i++ {
		sliceA[i] = float32(i) * 3.0
		sliceB[i] = float32(i) * 5.0
	}

	cfg := driver.LaunchConfig{
		GridDimX:  (n + 255) / 256,
		BlockDimX: 256,
	}

	if err := fn.Launch(cfg, mA, mB, mC, int32(n)); err != nil {
		t.Fatalf("fn.Launch failed: %v", err)
	}

	if err := ctx.Synchronize(); err != nil {
		t.Fatalf("ctx.Synchronize failed: %v", err)
	}

	for i := 0; i < n; i++ {
		expected := sliceA[i] + sliceB[i]
		if math.Abs(float64(sliceC[i]-expected)) > 1e-4 {
			t.Fatalf("mismatch at %d: got %f, expected %f", i, sliceC[i], expected)
		}
	}
	t.Logf("Successfully executed kernel compiled and linked via dynamic JIT Linker!")
}
