//go:build windows || linux

package driver_test

import (
	"testing"
	"unsafe"

	"github.com/DanielcoderX/cugo/driver"
)


func TestMemsetAndStreamWaitEvent(t *testing.T) {
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

	// 1. Test Context.MemsetD8
	const numBytes = 1024
	dMem8, err := ctx.Alloc(numBytes)
	if err != nil {
		t.Fatalf("Alloc dMem8 failed: %v", err)
	}
	defer ctx.Free(dMem8)

	if err := ctx.MemsetD8(dMem8, 0xAB, numBytes); err != nil {
		t.Fatalf("MemsetD8 failed: %v", err)
	}

	hBytes := make([]byte, numBytes)
	if err := ctx.CopyDtoH(hBytes, dMem8); err != nil {
		t.Fatalf("CopyDtoH dMem8 failed: %v", err)
	}
	for i, b := range hBytes {
		if b != 0xAB {
			t.Fatalf("MemsetD8 mismatch at %d: got 0x%X, expected 0xAB", i, b)
		}
	}
	t.Log("PASS: MemsetD8 verified")

	// 2. Test Stream.MemsetD32Async
	stream1, err := ctx.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream 1 failed: %v", err)
	}
	defer stream1.Destroy()

	stream2, err := ctx.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream 2 failed: %v", err)
	}
	defer stream2.Destroy()

	const numWords = 512
	const wordsBytes = numWords * 4
	dMem32, err := ctx.Alloc(wordsBytes)
	if err != nil {
		t.Fatalf("Alloc dMem32 failed: %v", err)
	}
	defer ctx.Free(dMem32)

	const magicVal = uint32(0xCAFEBABE)
	if err := stream1.MemsetD32Async(dMem32, magicVal, numWords); err != nil {
		t.Fatalf("MemsetD32Async failed: %v", err)
	}

	// 3. Test Stream.WaitEvent:
	// Record event in stream1 after Memset completes.
	// Stream2 waits on event, then executes.
	event, err := ctx.CreateEvent()
	if err != nil {
		t.Fatalf("CreateEvent failed: %v", err)
	}
	defer event.Destroy()

	if err := event.Record(stream1); err != nil {
		t.Fatalf("event.Record failed: %v", err)
	}

	// Stream2 waits on event from stream1
	if err := stream2.WaitEvent(event); err != nil {
		t.Fatalf("stream2.WaitEvent failed: %v", err)
	}

	// Stream2 copies data to host
	hWords := make([]uint32, numWords)
	wordsRaw := unsafe.Slice((*byte)(unsafe.Pointer(&hWords[0])), wordsBytes)
	if err := stream2.CopyDtoHAsync(wordsRaw, dMem32); err != nil {
		t.Fatalf("CopyDtoHAsync stream2 failed: %v", err)
	}

	if err := stream2.Synchronize(); err != nil {
		t.Fatalf("stream2.Synchronize failed: %v", err)
	}

	for i, w := range hWords {
		if w != magicVal {
			t.Fatalf("MemsetD32 mismatch at word %d: got 0x%X, expected 0x%X", i, w, magicVal)
		}
	}
	t.Log("PASS: MemsetD32Async + Stream.WaitEvent cross-stream synchronization verified")
}
