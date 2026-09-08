//go:build windows || linux

package driver_test

import (
	"math"
	"runtime"
	"testing"
	"unsafe"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/internal/nvapi"
	"github.com/cugo/cugo/kernels/scale"
	"github.com/cugo/cugo/kernels/vecadd"
)

func float32SliceToBytes(s []float32) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*4)
}

func bytesToFloat32Slice(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

func TestVecAddKernelLaunch(t *testing.T) {
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
		t.Fatalf("dev.CreateContext failed: %v", err)
	}
	defer ctx.Destroy()

	// Load PTX module
	mod, err := ctx.LoadModuleData(vecadd.PTX)
	if err != nil {
		t.Fatalf("ctx.LoadModuleData failed: %v", err)
	}
	defer mod.Unload()

	// Get function
	fn, err := mod.Function("vecAdd")
	if err != nil {
		t.Fatalf("mod.Function(\"vecAdd\") failed: %v", err)
	}
	if fn.Name() != "vecAdd" {
		t.Fatalf("expected fn name 'vecAdd', got %q", fn.Name())
	}

	// Prepare data
	const n = 50000
	const byteSize = n * 4

	hA := make([]float32, n)
	hB := make([]float32, n)
	hC := make([]float32, n)

	for i := 0; i < n; i++ {
		hA[i] = float32(i) * 1.5
		hB[i] = float32(i) * 2.5
	}

	// Alloc GPU buffers
	dA, err := ctx.Alloc(byteSize)
	if err != nil {
		t.Fatalf("ctx.Alloc(dA) failed: %v", err)
	}
	defer ctx.Free(dA)

	dB, err := ctx.Alloc(byteSize)
	if err != nil {
		t.Fatalf("ctx.Alloc(dB) failed: %v", err)
	}
	defer ctx.Free(dB)

	dC, err := ctx.Alloc(byteSize)
	if err != nil {
		t.Fatalf("ctx.Alloc(dC) failed: %v", err)
	}
	defer ctx.Free(dC)

	// Copy HtoD
	if err := ctx.CopyHtoD(dA, float32SliceToBytes(hA)); err != nil {
		t.Fatalf("ctx.CopyHtoD(dA) failed: %v", err)
	}
	if err := ctx.CopyHtoD(dB, float32SliceToBytes(hB)); err != nil {
		t.Fatalf("ctx.CopyHtoD(dB) failed: %v", err)
	}

	// Launch kernel
	blockDim := uint32(256)
	gridDim := (uint32(n) + blockDim - 1) / blockDim

	cfg := driver.LaunchConfig{
		GridDimX:  gridDim,
		BlockDimX: blockDim,
	}

	t.Logf("Launching kernel with Grid (%d, 1, 1), Block (%d, 1, 1), N=%d", gridDim, blockDim, n)
	if err := fn.Launch(cfg, dA, dB, dC, int32(n)); err != nil {
		t.Fatalf("fn.Launch failed: %v", err)
	}

	// Copy DtoH
	cBytes := make([]byte, byteSize)
	if err := ctx.CopyDtoH(cBytes, dC); err != nil {
		t.Fatalf("ctx.CopyDtoH(dC) failed: %v", err)
	}
	hC = bytesToFloat32Slice(cBytes)

	// Verify results
	for i := 0; i < n; i++ {
		expected := hA[i] + hB[i]
		if math.Abs(float64(hC[i]-expected)) > 1e-4 {
			t.Fatalf("result mismatch at index %d: got %f, expected %f", i, hC[i], expected)
		}
	}
	t.Logf("Successfully verified %d elements computed on GPU!", n)
}

func TestStreamsAndEvents(t *testing.T) {
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
		t.Fatalf("dev.CreateContext failed: %v", err)
	}
	defer ctx.Destroy()

	stream, err := ctx.CreateStream()
	if err != nil {
		t.Fatalf("ctx.CreateStream failed: %v", err)
	}
	defer stream.Destroy()

	startEv, err := ctx.CreateEvent()
	if err != nil {
		t.Fatalf("ctx.CreateEvent failed: %v", err)
	}
	defer startEv.Destroy()

	endEv, err := ctx.CreateEvent()
	if err != nil {
		t.Fatalf("ctx.CreateEvent failed: %v", err)
	}
	defer endEv.Destroy()

	// Record start
	if err := startEv.Record(stream); err != nil {
		t.Fatalf("startEv.Record failed: %v", err)
	}

	// Transfer 4MB asynchronously
	const size = 4 * 1024 * 1024
	buf := make([]byte, size)
	dptr, err := ctx.Alloc(size)
	if err != nil {
		t.Fatalf("ctx.Alloc failed: %v", err)
	}
	defer ctx.Free(dptr)

	if err := stream.CopyHtoDAsync(dptr, buf); err != nil {
		t.Fatalf("stream.CopyHtoDAsync failed: %v", err)
	}

	// Record end
	if err := endEv.Record(stream); err != nil {
		t.Fatalf("endEv.Record failed: %v", err)
	}

	// Synchronize
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream.Synchronize failed: %v", err)
	}

	ms, err := driver.ElapsedTime(startEv, endEv)
	if err != nil {
		t.Fatalf("driver.ElapsedTime failed: %v", err)
	}
	t.Logf("Async copy of %d bytes took: %.3f ms", size, ms)
	if ms < 0 {
		t.Errorf("expected non-negative elapsed time, got %f", ms)
	}
}

type ScaleParams struct {
	Factor float32
	Offset int32
}

func TestStructArgumentKernelLaunch(t *testing.T) {
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
		t.Fatal(err)
	}
	defer ctx.Destroy()

	mod, err := ctx.LoadModuleData(scale.PTX)
	if err != nil {
		t.Fatalf("ctx.LoadModuleData failed: %v", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("scaleKernel")
	if err != nil {
		t.Fatalf("mod.Function failed: %v", err)
	}

	const n = 1024
	const byteSize = n * 4

	hIn := make([]float32, n)
	for i := range hIn {
		hIn[i] = float32(i)
	}

	dIn, err := ctx.Alloc(byteSize); if err != nil { t.Fatal(err) }
	defer ctx.Free(dIn)
	dOut, err := ctx.Alloc(byteSize); if err != nil { t.Fatal(err) }
	defer ctx.Free(dOut)

	if err := ctx.CopyHtoD(dIn, float32SliceToBytes(hIn)); err != nil {
		t.Fatal(err)
	}

	// Pass struct by-value
	params := ScaleParams{
		Factor: 3.5,
		Offset: 42,
	}

	cfg := driver.LaunchConfig{
		GridDimX:  (n + 255) / 256,
		BlockDimX: 256,
	}

	// 1. Launch with struct passed by value
	if err := fn.Launch(cfg, dIn, dOut, params, int32(n)); err != nil {
		t.Fatalf("fn.Launch with struct by-value failed: %v", err)
	}

	hOut := make([]float32, n)
	if err := ctx.CopyDtoH(float32SliceToBytes(hOut), dOut); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < n; i++ {
		expected := hIn[i]*params.Factor + float32(params.Offset)
		if math.Abs(float64(hOut[i]-expected)) > 1e-4 {
			t.Fatalf("by-value struct mismatch at %d: got %f, expected %f", i, hOut[i], expected)
		}
	}
	t.Log("Successfully verified kernel with struct passed by value!")

	// 2. Launch with pointer to struct
	params.Factor = 10.0
	params.Offset = 5
	if err := fn.Launch(cfg, dIn, dOut, &params, int32(n)); err != nil {
		t.Fatalf("fn.Launch with pointer to struct failed: %v", err)
	}

	if err := ctx.CopyDtoH(float32SliceToBytes(hOut), dOut); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < n; i++ {
		expected := hIn[i]*params.Factor + float32(params.Offset)
		if math.Abs(float64(hOut[i]-expected)) > 1e-4 {
			t.Fatalf("pointer to struct mismatch at %d: got %f, expected %f", i, hOut[i], expected)
		}
	}
	t.Log("Successfully verified kernel with pointer to struct!")
}

func TestKernelArgTypes(t *testing.T) {
	// Test primitive constructors and conversions
	b := driver.Bool(true)
	if b == nil {
		t.Fatal("expected non-nil Bool arg")
	}

	i8 := driver.Int8(-42)
	if i8 == nil {
		t.Fatal("expected non-nil Int8 arg")
	}

	u8 := driver.Uint8(255)
	if u8 == nil {
		t.Fatal("expected non-nil Uint8 arg")
	}

	i16 := driver.Int16(-1000)
	if i16 == nil {
		t.Fatal("expected non-nil Int16 arg")
	}

	u16 := driver.Uint16(50000)
	if u16 == nil {
		t.Fatal("expected non-nil Uint16 arg")
	}
}

