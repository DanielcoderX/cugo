package tensor_test

import (
	"runtime"
	"testing"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/internal/nvapi"
	"github.com/cugo/cugo/tensor"
)

func TestTensorLifecycleAndSlicing(t *testing.T) {
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

	// 1. Create a 2D tensor [4, 4] with data 0..15
	data := make([]float32, 16)
	for i := range data {
		data[i] = float32(i)
	}

	tA, err := tensor.NewFromFloat32(ctx, data, 4, 4)
	if err != nil {
		t.Fatalf("tensor.NewFromFloat32 failed: %v", err)
	}
	defer tA.Close()

	if tA.Numel() != 16 {
		t.Fatalf("expected 16 elements, got %d", tA.Numel())
	}
	if !tA.IsContiguous() {
		t.Fatal("expected newly created tensor to be contiguous")
	}

	// 2. View/Reshape [4, 4] -> [2, 8]
	tView, err := tA.View(2, 8)
	if err != nil {
		t.Fatalf("tA.View failed: %v", err)
	}
	defer tView.Close()

	if tView.Shape()[0] != 2 || tView.Shape()[1] != 8 {
		t.Fatalf("unexpected view shape: %v", tView.Shape())
	}

	// 3. View with inferred dimension (-1) -> [16]
	tFlat, err := tA.View(-1)
	if err != nil {
		t.Fatalf("tA.View(-1) failed: %v", err)
	}
	defer tFlat.Close()

	if tFlat.Shape()[0] != 16 {
		t.Fatalf("expected shape [16], got %v", tFlat.Shape())
	}

	// 4. Zero-copy slice row 2 to 4 (rows 2 and 3)
	// row 2: [8, 9, 10, 11], row 3: [12, 13, 14, 15]
	tSlice, err := tA.Slice(0, 2, 4)
	if err != nil {
		t.Fatalf("tA.Slice failed: %v", err)
	}
	defer tSlice.Close()

	if tSlice.Shape()[0] != 2 || tSlice.Shape()[1] != 4 {
		t.Fatalf("unexpected slice shape: %v", tSlice.Shape())
	}

	cpuSlice, err := tSlice.ToCPUFloat32()
	if err != nil {
		t.Fatalf("tSlice.ToCPUFloat32 failed: %v", err)
	}

	if len(cpuSlice) != 8 {
		t.Fatalf("expected 8 slice elements, got %d", len(cpuSlice))
	}
	for i, v := range cpuSlice {
		expected := float32(8 + i)
		if v != expected {
			t.Fatalf("slice element %d mismatch: got %f, expected %f", i, v, expected)
		}
	}

	t.Logf("Successfully verified PyTorch-style Tensor View, Slice, and Device Transfers on GPU!")
}
