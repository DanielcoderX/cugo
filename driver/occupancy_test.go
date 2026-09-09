//go:build windows || linux

package driver_test

import (
	"runtime"
	"testing"

	"github.com/DanielcoderX/cugo/driver"
	"github.com/DanielcoderX/cugo/internal/nvapi"
	"github.com/DanielcoderX/cugo/kernels/vecadd"
)

func TestOccupancyCalculation(t *testing.T) {
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

	mod, err := ctx.LoadModuleData(vecadd.PTX)
	if err != nil {
		t.Fatalf("LoadModuleData failed: %v", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("vecAdd")
	if err != nil {
		t.Fatalf("Function vecAdd failed: %v", err)
	}

	// 1. Max active blocks for blockSize=256
	activeBlocks, err := fn.MaxActiveBlocksPerMultiprocessor(256, 0)
	if err != nil {
		t.Fatalf("MaxActiveBlocksPerMultiprocessor failed: %v", err)
	}
	if activeBlocks <= 0 {
		t.Fatalf("expected positive active blocks, got %d", activeBlocks)
	}
	t.Logf("Active blocks per SM (blockSize=256): %d", activeBlocks)

	// 2. Suggest optimal block size
	minGrid, optimalBlock, err := fn.SuggestBlockSize(0, 0)
	if err != nil {
		t.Fatalf("SuggestBlockSize failed: %v", err)
	}
	if minGrid <= 0 || optimalBlock <= 0 {
		t.Fatalf("invalid suggestion: minGrid=%d, optimalBlock=%d", minGrid, optimalBlock)
	}
	t.Logf("Suggested launch configuration: minGridSize=%d, optimalBlockSize=%d", minGrid, optimalBlock)
}
