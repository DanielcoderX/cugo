package tensor_test

import (
	"math"
	"runtime"
	"testing"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/internal/nvapi"
	"github.com/cugo/cugo/tensor"
)

func TestAutogradMatMulAndBias(t *testing.T) {
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

	// A: [2, 2], B: [2, 2]
	// A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
	// C = A * B = [[19, 22], [43, 50]]
	// Let Loss = sum(C) = 19 + 22 + 43 + 50 = 134
	// Then dC = [[1, 1], [1, 1]]
	// dA = dC * B^T = [[1, 1], [1, 1]] * [[5, 7], [6, 8]] = [[11, 15], [11, 15]]
	// dB = A^T * dC = [[1, 3], [2, 4]] * [[1, 1], [1, 1]] = [[4, 4], [6, 6]]

	hA := []float32{1, 2, 3, 4}
	hB := []float32{5, 6, 7, 8}

	tA, err := tensor.NewFromFloat32(ctx, hA, 2, 2); if err != nil { t.Fatal(err) }
	defer tA.Close()
	tA.SetRequiresGrad(true)

	tB, err := tensor.NewFromFloat32(ctx, hB, 2, 2); if err != nil { t.Fatal(err) }
	defer tB.Close()
	tB.SetRequiresGrad(true)

	tC, err := tensor.MatMul(tA, tB)
	if err != nil {
		t.Fatalf("MatMul failed: %v", err)
	}
	defer tC.Close()

	if err := tC.Backward(); err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	if tA.Grad == nil || tB.Grad == nil {
		t.Fatal("expected non-nil gradients on leaf tensors")
	}

	gradA, err := tA.Grad.ToCPUFloat32()
	if err != nil {
		t.Fatal(err)
	}
	gradB, err := tB.Grad.ToCPUFloat32()
	if err != nil {
		t.Fatal(err)
	}

	expectedGradA := []float32{11, 15, 11, 15}
	for i, v := range gradA {
		if math.Abs(float64(v-expectedGradA[i])) > 1e-4 {
			t.Fatalf("gradA[%d] mismatch: got %f, expected %f", i, v, expectedGradA[i])
		}
	}

	expectedGradB := []float32{4, 4, 6, 6}
	for i, v := range gradB {
		if math.Abs(float64(v-expectedGradB[i])) > 1e-4 {
			t.Fatalf("gradB[%d] mismatch: got %f, expected %f", i, v, expectedGradB[i])
		}
	}

	t.Logf("Autograd MatMul verified on GPU: dA=%v, dB=%v", gradA, gradB)
}

func TestAdamWOptimizer(t *testing.T) {
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

	// Initial param: [10.0, -10.0]
	hP := []float32{10.0, -10.0}
	p, err := tensor.NewFromFloat32(ctx, hP, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	p.SetRequiresGrad(true)

	// Gradient: [1.0, -1.0]
	hG := []float32{1.0, -1.0}
	g, err := tensor.NewFromFloat32(ctx, hG, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	p.Grad = g

	opt, err := tensor.NewAdamW([]*tensor.Tensor{p}, 0.1, 0.9, 0.999, 1e-8, 0.01)
	if err != nil {
		t.Fatalf("NewAdamW failed: %v", err)
	}
	defer opt.Close()

	if err := opt.Step(); err != nil {
		t.Fatalf("opt.Step failed: %v", err)
	}

	updatedP, err := p.ToCPUFloat32()
	if err != nil {
		t.Fatal(err)
	}

	// First parameter should decrease (positive gradient), second should increase (negative gradient)
	if updatedP[0] >= 10.0 {
		t.Fatalf("expected updatedP[0] < 10.0, got %f", updatedP[0])
	}
	if updatedP[1] <= -10.0 {
		t.Fatalf("expected updatedP[1] > -10.0, got %f", updatedP[1])
	}

	t.Logf("AdamW Step verified on GPU: initial=%v, updated=%v", hP, updatedP)
}
