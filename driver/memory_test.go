//go:build windows || linux

package driver_test

import (
	"bytes"
	"crypto/rand"
	"errors"
	"testing"

	"github.com/DanielcoderX/cugo/driver"
	"github.com/DanielcoderX/cugo/internal/nvapi"
)

func TestContextAndMemoryRoundTrip(t *testing.T) {
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
	defer func() {
		if err := ctx.Destroy(); err != nil && !errors.Is(err, driver.ErrContextDestroyed) {
			t.Errorf("ctx.Destroy failed: %v", err)
		}
	}()

	// Test 1MB round-trip
	const size = 1024 * 1024
	src := make([]byte, size)
	if _, err := rand.Read(src); err != nil {
		t.Fatalf("rand.Read failed: %v", err)
	}

	dptr, err := ctx.Alloc(size)
	if err != nil {
		t.Fatalf("ctx.Alloc failed: %v", err)
	}
	if dptr.IsNil() {
		t.Fatal("expected non-nil DevicePtr")
	}
	t.Logf("Allocated %d bytes at device address 0x%x", size, dptr.Uintptr())

	if err := ctx.CopyHtoD(dptr, src); err != nil {
		t.Fatalf("ctx.CopyHtoD failed: %v", err)
	}

	dst := make([]byte, size)
	if err := ctx.CopyDtoH(dst, dptr); err != nil {
		t.Fatalf("ctx.CopyDtoH failed: %v", err)
	}

	if !bytes.Equal(src, dst) {
		t.Fatal("memory round-trip mismatch: data corrupted on GPU transfer")
	}
	t.Logf("Successfully verified %d bytes round-trip HtoD -> DtoH", size)

	if err := ctx.Free(dptr); err != nil {
		t.Fatalf("ctx.Free failed: %v", err)
	}
}

func TestMemorySafety(t *testing.T) {
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

	// 1. Zero-size alloc
	_, err = ctx.Alloc(0)
	if !errors.Is(err, driver.ErrZeroSizeAlloc) {
		t.Fatalf("expected ErrZeroSizeAlloc, got: %v", err)
	}

	// 2. Free null pointer
	err = ctx.Free(0)
	if !errors.Is(err, driver.ErrNullPointer) {
		t.Fatalf("expected ErrNullPointer, got: %v", err)
	}

	// 3. Destroy context
	if err := ctx.Destroy(); err != nil {
		t.Fatalf("ctx.Destroy failed: %v", err)
	}

	// 4. Double destroy
	if err := ctx.Destroy(); !errors.Is(err, driver.ErrContextDestroyed) {
		t.Fatalf("expected ErrContextDestroyed on double destroy, got: %v", err)
	}

	// 5. Alloc on destroyed context
	_, err = ctx.Alloc(1024)
	if !errors.Is(err, driver.ErrContextDestroyed) {
		t.Fatalf("expected ErrContextDestroyed on Alloc, got: %v", err)
	}

	// 6. SetCurrent on destroyed context
	if err := ctx.SetCurrent(); !errors.Is(err, driver.ErrContextDestroyed) {
		t.Fatalf("expected ErrContextDestroyed on SetCurrent, got: %v", err)
	}
}
