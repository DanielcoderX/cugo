//go:build windows || linux

package driver_test

import (
	"testing"
	"unsafe"

	"github.com/DanielcoderX/cugo/driver"
	"github.com/DanielcoderX/cugo/kernels/vecadd"
)

func TestStreamPriority(t *testing.T) {
	if err := driver.Init(); err != nil {
		t.Fatalf("driver.Init failed: %v", err)
	}

	count, err := driver.DeviceCount()
	if err != nil || count == 0 {
		t.Skip("no CUDA devices available")
	}

	dev, err := driver.GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice(0) failed: %v", err)
	}

	ctx, err := dev.CreateContext()
	if err != nil {
		t.Fatalf("CreateContext failed: %v", err)
	}
	defer ctx.Destroy()

	least, greatest, err := ctx.StreamPriorityRange()
	if err != nil {
		t.Fatalf("StreamPriorityRange failed: %v", err)
	}
	t.Logf("Stream Priority Range: least=%d, greatest=%d", least, greatest)

	// Create high priority stream
	highStream, err := ctx.CreateStreamWithPriority(greatest)
	if err != nil {
		t.Fatalf("CreateStreamWithPriority(greatest=%d) failed: %v", greatest, err)
	}
	defer highStream.Destroy()

	if highStream.Priority() != greatest {
		t.Fatalf("expected priority %d, got %d", greatest, highStream.Priority())
	}

	// Create low priority stream
	lowStream, err := ctx.CreateStreamWithPriority(least)
	if err != nil {
		t.Fatalf("CreateStreamWithPriority(least=%d) failed: %v", least, err)
	}
	defer lowStream.Destroy()

	if lowStream.Priority() != least {
		t.Fatalf("expected priority %d, got %d", least, lowStream.Priority())
	}

	// Execute kernel on high priority stream
	mod, err := ctx.LoadModuleData(vecadd.PTX)
	if err != nil {
		t.Fatalf("LoadModuleData failed: %v", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("vecAdd")
	if err != nil {
		t.Fatalf("mod.Function failed: %v", err)
	}

	const n = 1024
	const sizeBytes = n * 4

	hA := make([]float32, n)
	hB := make([]float32, n)
	for i := range hA {
		hA[i] = 1.5
		hB[i] = 2.5
	}

	dA, err := ctx.Alloc(sizeBytes)
	if err != nil {
		t.Fatalf("Alloc dA failed: %v", err)
	}
	defer ctx.Free(dA)

	dB, err := ctx.Alloc(sizeBytes)
	if err != nil {
		t.Fatalf("Alloc dB failed: %v", err)
	}
	defer ctx.Free(dB)

	dC, err := ctx.Alloc(sizeBytes)
	if err != nil {
		t.Fatalf("Alloc dC failed: %v", err)
	}
	defer ctx.Free(dC)

	if err := highStream.CopyHtoDAsync(dA, unsafe.Slice((*byte)(unsafe.Pointer(&hA[0])), sizeBytes)); err != nil {
		t.Fatalf("CopyHtoDAsync dA failed: %v", err)
	}
	if err := highStream.CopyHtoDAsync(dB, unsafe.Slice((*byte)(unsafe.Pointer(&hB[0])), sizeBytes)); err != nil {
		t.Fatalf("CopyHtoDAsync dB failed: %v", err)
	}

	cfg := driver.LaunchConfig{
		GridDimX:  4,
		BlockDimX: 256,
		Stream:    highStream,
	}
	if err := fn.Launch(cfg, dA, dB, dC, int32(n)); err != nil {
		t.Fatalf("fn.Launch failed: %v", err)
	}

	hC := make([]float32, n)
	if err := highStream.CopyDtoHAsync(unsafe.Slice((*byte)(unsafe.Pointer(&hC[0])), sizeBytes), dC); err != nil {
		t.Fatalf("CopyDtoHAsync dC failed: %v", err)
	}

	if err := highStream.Synchronize(); err != nil {
		t.Fatalf("highStream.Synchronize failed: %v", err)
	}

	for i, v := range hC {
		if v != 4.0 {
			t.Fatalf("highStream result mismatch at %d: got %f, expected 4.0", i, v)
		}
	}
	t.Log("PASS: High priority stream executed and synchronized successfully")
}
