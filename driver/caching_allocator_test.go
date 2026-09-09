//go:build windows || linux

package driver_test

import (
	"runtime"
	"testing"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/internal/nvapi"
)

func TestCachingAllocator(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := nvapi.CheckDriver(); err != nil {
		t.Skip("no CUDA driver")
	}
	if err := driver.Init(); err != nil {
		t.Skip("driver.Init failed")
	}
	devs, err := driver.Devices()
	if err != nil || len(devs) == 0 {
		t.Skip("no devices")
	}

	ctx, err := devs[0].CreateContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Destroy()

	alloc := driver.NewCachingAllocator(ctx)
	defer alloc.EmptyCache()

	// 1. Initial allocation
	const size = 1024 * 1024 // 1 MB
	p1, err := alloc.Alloc(size)
	if err != nil {
		t.Fatalf("alloc.Alloc p1 failed: %v", err)
	}
	if p1 == 0 {
		t.Fatal("expected non-zero pointer")
	}

	stats1 := alloc.Stats()
	if stats1.ActiveBlocks != 1 || stats1.CachedBlocks != 0 {
		t.Fatalf("unexpected stats after alloc: %+v", stats1)
	}

	// 2. Free block into cache
	if err := alloc.Free(p1); err != nil {
		t.Fatalf("alloc.Free failed: %v", err)
	}

	stats2 := alloc.Stats()
	if stats2.ActiveBlocks != 0 || stats2.CachedBlocks != 1 {
		t.Fatalf("unexpected stats after free: %+v", stats2)
	}

	// 3. Re-allocate same size: must reuse exact same device pointer without driver alloc!
	p2, err := alloc.Alloc(size)
	if err != nil {
		t.Fatalf("alloc.Alloc p2 failed: %v", err)
	}
	if p2 != p1 {
		t.Fatalf("expected reused pointer 0x%x, got 0x%x", p1, p2)
	}

	stats3 := alloc.Stats()
	if stats3.ActiveBlocks != 1 || stats3.CachedBlocks != 0 {
		t.Fatalf("unexpected stats after reuse: %+v", stats3)
	}

	// 4. Clean up
	if err := alloc.Free(p2); err != nil {
		t.Fatal(err)
	}
	if err := alloc.EmptyCache(); err != nil {
		t.Fatal(err)
	}

	statsFinal := alloc.Stats()
	if statsFinal.ReservedBytes != 0 || statsFinal.CachedBlocks != 0 {
		t.Fatalf("expected 0 reserved bytes after EmptyCache, got %+v", statsFinal)
	}
}
