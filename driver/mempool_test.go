//go:build windows || linux

package driver_test

import (
	"bytes"
	"runtime"
	"testing"

	"github.com/DanielcoderX/cugo/driver"
	"github.com/DanielcoderX/cugo/internal/nvapi"
)

func TestStreamOrderedAllocAndMemPool(t *testing.T) {
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
		t.Fatalf("CreateContext failed: %v", err)
	}
	defer ctx.Destroy()

	stream, err := ctx.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream failed: %v", err)
	}
	defer stream.Destroy()

	const size = 2 * 1024 * 1024 // 2MB
	// 1. Allocate asynchronously on stream
	dptr, err := ctx.AllocAsync(size, stream)
	if err != nil {
		t.Fatalf("ctx.AllocAsync failed: %v", err)
	}
	if dptr == 0 {
		t.Fatal("expected non-zero device pointer")
	}

	// 2. Async transfer
	src := make([]byte, size)
	for i := range src {
		src[i] = byte((i * 17) & 0xFF)
	}
	dst := make([]byte, size)

	if err := stream.CopyHtoDAsync(dptr, src); err != nil {
		t.Fatalf("CopyHtoDAsync failed: %v", err)
	}
	if err := stream.CopyDtoHAsync(dst, dptr); err != nil {
		t.Fatalf("CopyDtoHAsync failed: %v", err)
	}

	// 3. Free asynchronously on stream (stream-ordered free executes only after copy completes!)
	if err := ctx.FreeAsync(dptr, stream); err != nil {
		t.Fatalf("ctx.FreeAsync failed: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream.Synchronize failed: %v", err)
	}

	// Verify data correctness
	if !bytes.Equal(src, dst) {
		t.Fatal("data mismatch in stream-ordered async transfer")
	}
	t.Logf("Stream-ordered async allocation round-trip of %d bytes PASSED", size)

	// 4. Inspect DefaultMemPool
	pool, err := dev.DefaultMemPool()
	if err != nil {
		t.Fatalf("dev.DefaultMemPool failed: %v", err)
	}
	reserved, err := pool.ReservedMemCurrent()
	if err != nil {
		t.Fatalf("pool.ReservedMemCurrent failed: %v", err)
	}
	used, err := pool.UsedMemCurrent()
	if err != nil {
		t.Fatalf("pool.UsedMemCurrent failed: %v", err)
	}
	t.Logf("MemPool: reserved=%d bytes, used=%d bytes", reserved, used)

	// 5. Trim pool
	if err := pool.TrimTo(0); err != nil {
		t.Fatalf("pool.TrimTo failed: %v", err)
	}
}
