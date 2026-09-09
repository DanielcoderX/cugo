//go:build windows

package gemm_test

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
	"unsafe"

	"github.com/DanielcoderX/cugo/driver"
	"github.com/DanielcoderX/cugo/internal/nvapi"
	"github.com/DanielcoderX/cugo/kernels/gemm"
)

func TestGEMMKernel(t *testing.T) {
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

	mod, err := ctx.LoadModuleData(gemm.PTX)
	if err != nil {
		t.Fatalf("LoadModuleData failed: %v", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("gemm")
	if err != nil {
		t.Fatalf("Function gemm failed: %v", err)
	}

	const M = 128
	const N = 128
	const K = 128

	hA := make([]float32, M*K)
	hB := make([]float32, K*N)
	hC := make([]float32, M*N)
	expectedC := make([]float32, M*N)

	r := rand.New(rand.NewSource(42))
	for i := range hA {
		hA[i] = r.Float32()
	}
	for i := range hB {
		hB[i] = r.Float32()
	}

	// Compute CPU reference
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += hA[i*K+k] * hB[k*N+j]
			}
			expectedC[i*N+j] = sum
		}
	}

	sizeA := uint64(len(hA) * 4)
	sizeB := uint64(len(hB) * 4)
	sizeC := uint64(len(hC) * 4)

	dA, err := ctx.Alloc(sizeA); if err != nil { t.Fatal(err) }
	defer ctx.Free(dA)
	dB, err := ctx.Alloc(sizeB); if err != nil { t.Fatal(err) }
	defer ctx.Free(dB)
	dC, err := ctx.Alloc(sizeC); if err != nil { t.Fatal(err) }
	defer ctx.Free(dC)

	rawA := unsafe.Slice((*byte)(unsafe.Pointer(&hA[0])), sizeA)
	rawB := unsafe.Slice((*byte)(unsafe.Pointer(&hB[0])), sizeB)
	rawC := unsafe.Slice((*byte)(unsafe.Pointer(&hC[0])), sizeC)

	if err := ctx.CopyHtoD(dA, rawA); err != nil { t.Fatal(err) }
	if err := ctx.CopyHtoD(dB, rawB); err != nil { t.Fatal(err) }

	cfg := driver.LaunchConfig{
		GridDimX:  uint32((N + 15) / 16),
		GridDimY:  uint32((M + 15) / 16),
		BlockDimX: 16,
		BlockDimY: 16,
	}

	if err := fn.Launch(cfg, dA, dB, dC, int32(M), int32(N), int32(K)); err != nil {
		t.Fatalf("fn.Launch failed: %v", err)
	}

	if err := ctx.Synchronize(); err != nil {
		t.Fatalf("ctx.Synchronize failed: %v", err)
	}

	if err := ctx.CopyDtoH(rawC, dC); err != nil {
		t.Fatalf("CopyDtoH failed: %v", err)
	}

	for i := 0; i < M*N; i++ {
		diff := math.Abs(float64(hC[i] - expectedC[i]))
		if diff > 1e-3 {
			t.Fatalf("GEMM mismatch at %d: got %f, expected %f (diff %f)", i, hC[i], expectedC[i], diff)
		}
	}

	t.Logf("Tiled GEMM (%dx%dx%d) verified successfully on GPU!", M, N, K)
}
