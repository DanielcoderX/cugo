//go:build windows

package main

import (
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"time"
	"unsafe"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/kernels/gemm"
)

func main() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := driver.Init(); err != nil {
		log.Fatalf("driver.Init failed: %v", err)
	}

	devs, err := driver.Devices()
	if err != nil || len(devs) == 0 {
		log.Fatal("no CUDA devices available")
	}
	dev := devs[0]
	name, _ := dev.Name()
	fmt.Printf("Running Tiled GEMM Benchmark on %s\n", name)

	ctx, err := dev.CreateContext()
	if err != nil {
		log.Fatalf("CreateContext failed: %v", err)
	}
	defer ctx.Destroy()

	mod, err := ctx.LoadModuleData(gemm.PTX)
	if err != nil {
		log.Fatalf("LoadModuleData failed: %v", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("gemm")
	if err != nil {
		log.Fatalf("Function gemm failed: %v", err)
	}

	// Matrix sizes: 1024 x 1024
	const M = 1024
	const N = 1024
	const K = 1024

	const bytesA = M * K * 4
	const bytesB = K * N * 4
	const bytesC = M * N * 4

	hA := make([]float32, M*K)
	hB := make([]float32, K*N)
	hC := make([]float32, M*N)

	r := rand.New(rand.NewSource(12345))
	for i := range hA {
		hA[i] = r.Float32()
	}
	for i := range hB {
		hB[i] = r.Float32()
	}

	dA, err := ctx.Alloc(bytesA); if err != nil { log.Fatal(err) }
	defer ctx.Free(dA)
	dB, err := ctx.Alloc(bytesB); if err != nil { log.Fatal(err) }
	defer ctx.Free(dB)
	dC, err := ctx.Alloc(bytesC); if err != nil { log.Fatal(err) }
	defer ctx.Free(dC)

	rawA := unsafe.Slice((*byte)(unsafe.Pointer(&hA[0])), bytesA)
	rawB := unsafe.Slice((*byte)(unsafe.Pointer(&hB[0])), bytesB)
	rawC := unsafe.Slice((*byte)(unsafe.Pointer(&hC[0])), bytesC)

	if err := ctx.CopyHtoD(dA, rawA); err != nil { log.Fatal(err) }
	if err := ctx.CopyHtoD(dB, rawB); err != nil { log.Fatal(err) }

	cfg := driver.LaunchConfig{
		GridDimX:  uint32((N + 15) / 16),
		GridDimY:  uint32((M + 15) / 16),
		BlockDimX: 16,
		BlockDimY: 16,
	}

	// Warmup
	for i := 0; i < 5; i++ {
		_ = fn.Launch(cfg, dA, dB, dC, int32(M), int32(N), int32(K))
	}
	_ = ctx.Synchronize()

	// Benchmark
	const iters = 20
	start := time.Now()
	for i := 0; i < iters; i++ {
		if err := fn.Launch(cfg, dA, dB, dC, int32(M), int32(N), int32(K)); err != nil {
			log.Fatalf("fn.Launch failed: %v", err)
		}
	}
	if err := ctx.Synchronize(); err != nil {
		log.Fatalf("ctx.Synchronize failed: %v", err)
	}
	totalElapsed := time.Since(start)
	avgDuration := totalElapsed / iters

	// FLOPs = 2 * M * N * K
	flops := float64(2) * float64(M) * float64(N) * float64(K)
	gflops := (flops / avgDuration.Seconds()) / 1e9

	if err := ctx.CopyDtoH(rawC, dC); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Matrix Size: %dx%d * %dx%d\n", M, K, K, N)
	fmt.Printf("Avg Kernel Latency: %v\n", avgDuration)
	fmt.Printf("Throughput: %.2f GFLOPS\n", gflops)
	fmt.Printf("Verification sample C[0]=%.4f, C[%d]=%.4f\n", hC[0], len(hC)-1, hC[len(hC)-1])
}
