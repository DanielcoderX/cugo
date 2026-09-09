package gemv_test

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/kernels/gemv"
)

func cpuGEMV(A, x, y []float32, m, n int, alpha, beta float32) []float32 {
	out := make([]float32, m)
	copy(out, y)
	for r := 0; r < m; r++ {
		sum := float32(0)
		rowStart := r * n
		for c := 0; c < n; c++ {
			sum += A[rowStart+c] * x[c]
		}
		if beta == 0 {
			out[r] = alpha * sum
		} else {
			out[r] = alpha * sum + beta * out[r]
		}
	}
	return out
}

func TestGEMVKernel(t *testing.T) {
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

	stream, err := ctx.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream failed: %v", err)
	}
	defer stream.Destroy()

	testCases := []struct {
		m, n  int
		alpha float32
		beta  float32
	}{
		{m: 32, n: 64, alpha: 1.0, beta: 0.0},
		{m: 128, n: 128, alpha: 1.5, beta: 0.5},
		{m: 256, n: 1024, alpha: 1.0, beta: 0.0},
		{m: 1024, n: 256, alpha: 2.0, beta: 1.0},
		{m: 2048, n: 4096, alpha: 1.0, beta: 0.0},
	}

	for _, tc := range testCases {
		aBytes := uint64(tc.m * tc.n * 4)
		xBytes := uint64(tc.n * 4)
		yBytes := uint64(tc.m * 4)

		hA := make([]float32, tc.m*tc.n)
		for i := range hA {
			hA[i] = (rand.Float32() - 0.5) * 2.0
		}
		hx := make([]float32, tc.n)
		for i := range hx {
			hx[i] = (rand.Float32() - 0.5) * 2.0
		}
		hy := make([]float32, tc.m)
		for i := range hy {
			hy[i] = (rand.Float32() - 0.5) * 2.0
		}

		cpuExpected := cpuGEMV(hA, hx, hy, tc.m, tc.n, tc.alpha, tc.beta)

		dA, err := ctx.Alloc(aBytes)
		if err != nil {
			t.Fatalf("Alloc dA failed: %v", err)
		}
		defer ctx.Free(dA)

		dX, err := ctx.Alloc(xBytes)
		if err != nil {
			t.Fatalf("Alloc dX failed: %v", err)
		}
		defer ctx.Free(dX)

		dY, err := ctx.Alloc(yBytes)
		if err != nil {
			t.Fatalf("Alloc dY failed: %v", err)
		}
		defer ctx.Free(dY)

		if err := ctx.CopyHtoD(dA, unsafe.Slice((*byte)(unsafe.Pointer(&hA[0])), aBytes)); err != nil {
			t.Fatalf("CopyHtoD dA failed: %v", err)
		}
		if err := ctx.CopyHtoD(dX, unsafe.Slice((*byte)(unsafe.Pointer(&hx[0])), xBytes)); err != nil {
			t.Fatalf("CopyHtoD dX failed: %v", err)
		}
		if err := ctx.CopyHtoD(dY, unsafe.Slice((*byte)(unsafe.Pointer(&hy[0])), yBytes)); err != nil {
			t.Fatalf("CopyHtoD dY failed: %v", err)
		}

		if err := gemv.ExecuteGEMV(ctx, stream, dA, dX, dY, tc.m, tc.n, tc.alpha, tc.beta); err != nil {
			t.Fatalf("ExecuteGEMV failed: %v", err)
		}

		if err := stream.Synchronize(); err != nil {
			t.Fatalf("stream.Synchronize failed: %v", err)
		}

		hOutput := make([]float32, tc.m)
		if err := ctx.CopyDtoH(unsafe.Slice((*byte)(unsafe.Pointer(&hOutput[0])), yBytes), dY); err != nil {
			t.Fatalf("CopyDtoH dY failed: %v", err)
		}

		for i := 0; i < tc.m; i++ {
			diff := math.Abs(float64(hOutput[i] - cpuExpected[i]))
			tol := math.Max(1e-3, math.Abs(float64(cpuExpected[i]))*1e-3)
			if diff > tol {
				t.Fatalf("Shape [%dx%d] element %d mismatch: GPU=%f, CPU=%f (diff=%f > tol=%f)",
					tc.m, tc.n, i, hOutput[i], cpuExpected[i], diff, tol)
			}
		}

		t.Logf("PASS: shape [%dx%d] (alpha=%.1f, beta=%.1f)", tc.m, tc.n, tc.alpha, tc.beta)
	}
}
