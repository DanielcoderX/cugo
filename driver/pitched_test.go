//go:build windows

package driver_test

import (
	"errors"
	"runtime"
	"testing"
	"unsafe"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/internal/nvapi"
)

func TestPitchedMemoryAndCopy2D(t *testing.T) {
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

	const rows = 64
	const cols = 128
	const elemSize = 4 // float32
	const widthInBytes = cols * elemSize

	// 1. Allocate pitched device memory
	pitched, err := ctx.AllocPitch(widthInBytes, rows, elemSize)
	if err != nil {
		t.Fatalf("AllocPitch failed: %v", err)
	}
	defer ctx.Free(pitched.Ptr)

	if pitched.Ptr == 0 {
		t.Fatal("expected non-zero device pointer")
	}
	if pitched.Pitch < widthInBytes {
		t.Fatalf("expected pitch >= width (%d), got %d", widthInBytes, pitched.Pitch)
	}
	t.Logf("Allocated 2D pitched memory: width=%d bytes, rows=%d, pitch=%d bytes", widthInBytes, rows, pitched.Pitch)

	// 2. Prepare 2D host matrix data
	srcMatrix := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			srcMatrix[r*cols+c] = float32(r*1000 + c)
		}
	}
	dstMatrix := make([]float32, rows*cols)

	// 3. Copy HtoD 2D
	copyHtoD := driver.Copy2DParams{
		SrcMemoryType: driver.MemTypeHost,
		SrcHost:       unsafe.Pointer(&srcMatrix[0]),
		SrcPitch:      widthInBytes,

		DstMemoryType: driver.MemTypeDevice,
		DstDevice:     pitched.Ptr,
		DstPitch:      pitched.Pitch,

		WidthInBytes: widthInBytes,
		Height:       rows,
	}
	if err := ctx.Copy2D(copyHtoD); err != nil {
		t.Fatalf("Copy2D (HtoD) failed: %v", err)
	}

	// 4. Copy DtoH 2D
	copyDtoH := driver.Copy2DParams{
		SrcMemoryType: driver.MemTypeDevice,
		SrcDevice:     pitched.Ptr,
		SrcPitch:      pitched.Pitch,

		DstMemoryType: driver.MemTypeHost,
		DstHost:       unsafe.Pointer(&dstMatrix[0]),
		DstPitch:      widthInBytes,

		WidthInBytes: widthInBytes,
		Height:       rows,
	}
	if err := ctx.Copy2D(copyDtoH); err != nil {
		t.Fatalf("Copy2D (DtoH) failed: %v", err)
	}

	// 5. Verify data integrity across the entire 2D matrix
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			idx := r*cols + c
			if srcMatrix[idx] != dstMatrix[idx] {
				t.Fatalf("mismatch at [%d,%d]: got %f, expected %f", r, c, dstMatrix[idx], srcMatrix[idx])
			}
		}
	}
	t.Logf("2D pitched copy round-trip verified successfully for %dx%d matrix!", rows, cols)

	// 6. Test CUDA 2D Array allocation and cleanup
	arr, err := ctx.CreateArray2D(cols, rows, driver.ArrayFormatFloat, 1)
	if err != nil {
		t.Fatalf("CreateArray2D failed: %v", err)
	}
	if err := arr.Destroy(); err != nil {
		t.Fatalf("arr.Destroy failed: %v", err)
	}
	if err := arr.Destroy(); !errors.Is(err, driver.ErrArrayDestroyed) {
		t.Fatalf("expected ErrArrayDestroyed, got: %v", err)
	}
	t.Logf("CUDA Array 2D allocation and destruction verified!")
}
