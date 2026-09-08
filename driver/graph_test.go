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

func TestCUDAGraphCaptureAndLaunch(t *testing.T) {
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

	mod, err := ctx.LoadModuleData(vecadd.PTX)
	if err != nil {
		t.Fatalf("LoadModuleData failed: %v", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("vecAdd")
	if err != nil {
		t.Fatalf("Function vecAdd failed: %v", err)
	}

	const n = 10000
	const byteSize = n * 4

	dA, err := ctx.Alloc(byteSize); if err != nil { t.Fatal(err) }
	defer ctx.Free(dA)
	dB, err := ctx.Alloc(byteSize); if err != nil { t.Fatal(err) }
	defer ctx.Free(dB)
	dC, err := ctx.Alloc(byteSize); if err != nil { t.Fatal(err) }
	defer ctx.Free(dC)

	hA := make([]float32, n)
	hB := make([]float32, n)
	hC := make([]float32, n)

	for i := 0; i < n; i++ {
		hA[i] = float32(i) * 1.5
		hB[i] = float32(i) * 2.5
	}

	rawA := unsafe.Slice((*byte)(unsafe.Pointer(&hA[0])), byteSize)
	rawB := unsafe.Slice((*byte)(unsafe.Pointer(&hB[0])), byteSize)
	rawC := unsafe.Slice((*byte)(unsafe.Pointer(&hC[0])), byteSize)

	// Copy inputs to device upfront
	if err := ctx.CopyHtoD(dA, rawA); err != nil { t.Fatal(err) }
	if err := ctx.CopyHtoD(dB, rawB); err != nil { t.Fatal(err) }

	stream, err := ctx.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream failed: %v", err)
	}
	defer stream.Destroy()

	// 1. Begin graph capture
	if err := stream.BeginCapture(); err != nil {
		t.Fatalf("BeginCapture failed: %v", err)
	}

	capturing, err := stream.IsCapturing()
	if err != nil || !capturing {
		t.Fatalf("expected stream to be capturing, got %v, err: %v", capturing, err)
	}

	cfg := driver.LaunchConfig{
		GridDimX:  (n + 255) / 256,
		BlockDimX: 256,
		Stream:    stream,
	}

	// Launch kernel inside stream capture (operation recorded into graph, not executed yet)
	if err := fn.Launch(cfg, dA, dB, dC, int32(n)); err != nil {
		t.Fatalf("fn.Launch inside capture failed: %v", err)
	}

	// 2. End graph capture
	graph, err := stream.EndCapture()
	if err != nil {
		t.Fatalf("EndCapture failed: %v", err)
	}
	defer graph.Destroy()

	capturingAfter, _ := stream.IsCapturing()
	if capturingAfter {
		t.Fatal("expected stream to no longer be capturing")
	}

	// 3. Instantiate executable graph
	graphExec, err := graph.Instantiate()
	if err != nil {
		t.Fatalf("graph.Instantiate failed: %v", err)
	}
	defer graphExec.Destroy()

	// 4. Launch executable graph on stream
	if err := graphExec.Launch(stream); err != nil {
		t.Fatalf("graphExec.Launch failed: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream.Synchronize failed: %v", err)
	}

	// 5. Copy result back and verify
	if err := ctx.CopyDtoH(rawC, dC); err != nil {
		t.Fatalf("CopyDtoH failed: %v", err)
	}

	for i := 0; i < n; i++ {
		expected := hA[i] + hB[i]
		if math.Abs(float64(hC[i]-expected)) > 1e-4 {
			t.Fatalf("mismatch at %d: got %f, expected %f", i, hC[i], expected)
		}
	}
	t.Logf("Successfully captured, instantiated, and launched CUDA Graph for %d elements!", n)

	// Double destroy checks
	if err := graphExec.Destroy(); err != nil {
		t.Fatalf("graphExec.Destroy failed: %v", err)
	}
	if err := graphExec.Destroy(); !errors.Is(err, driver.ErrGraphExecDestroyed) {
		t.Fatalf("expected ErrGraphExecDestroyed on double destroy, got: %v", err)
	}
}
