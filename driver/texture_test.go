//go:build windows || linux

package driver_test

import (
	"errors"
	"runtime"
	"testing"

	"github.com/DanielcoderX/cugo/driver"
	"github.com/DanielcoderX/cugo/internal/nvapi"
)

func TestProfilerAndTextureSurfaceObjects(t *testing.T) {
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

	// 1. Test Profiler Start and Stop inside active context
	if err := driver.ProfilerStart(); err != nil {
		t.Fatalf("driver.ProfilerStart failed: %v", err)
	}
	if err := driver.ProfilerStop(); err != nil {
		t.Fatalf("driver.ProfilerStop failed: %v", err)
	}
	t.Log("CUDA Profiler Start/Stop successful!")

	// 2. Create 2D Array
	arr, err := ctx.CreateArray2D(64, 64, driver.ArrayFormatFloat, 1)
	if err != nil {
		t.Fatalf("CreateArray2D failed: %v", err)
	}
	defer arr.Destroy()

	// 3. Create and Destroy Texture Object
	tex, err := ctx.CreateTextureObject(arr)
	if err != nil {
		t.Fatalf("CreateTextureObject failed: %v", err)
	}
	if tex.Handle() == 0 {
		t.Fatal("expected non-zero texture handle")
	}
	t.Logf("Created Texture Object handle: 0x%x", tex.Handle())

	if err := tex.Destroy(); err != nil {
		t.Fatalf("tex.Destroy failed: %v", err)
	}
	if err := tex.Destroy(); !errors.Is(err, driver.ErrTextureDestroyed) {
		t.Fatalf("expected ErrTextureDestroyed, got: %v", err)
	}

	// 4. Create and Destroy Surface Object
	surf, err := ctx.CreateSurfaceObject(arr)
	if err != nil {
		t.Fatalf("CreateSurfaceObject failed: %v", err)
	}
	if surf.Handle() == 0 {
		t.Fatal("expected non-zero surface handle")
	}
	t.Logf("Created Surface Object handle: 0x%x", surf.Handle())

	if err := surf.Destroy(); err != nil {
		t.Fatalf("surf.Destroy failed: %v", err)
	}
	if err := surf.Destroy(); !errors.Is(err, driver.ErrSurfaceDestroyed) {
		t.Fatalf("expected ErrSurfaceDestroyed, got: %v", err)
	}
}
