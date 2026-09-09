//go:build windows || linux

package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"
	"unsafe"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/kernels/gemm"
)

func main() {
	fmt.Println("================================================================================")
	fmt.Println(" cugo: Tensor Parallelism (TP) Column-Parallel Sharded GEMM (Zero CGO)")
	fmt.Println("================================================================================")

	if err := driver.Init(); err != nil {
		log.Fatalf("driver.Init failed: %v", err)
	}

	devs, err := driver.Devices()
	if err != nil || len(devs) == 0 {
		log.Fatal("no CUDA devices found")
	}

	numDevs := len(devs)
	fmt.Printf("Detected %d CUDA device(s)\n", numDevs)
	for i, dev := range devs {
		name, _ := dev.Name()
		fmt.Printf("  [GPU %d] %s\n", i, name)
	}
	fmt.Println()

	// Dimensions: M (batch/seq) = 256, K (hidden_in) = 512, N (hidden_out) = 1024
	const M = 256
	const K = 512
	const N = 1024
	const N1 = N / 2
	const N2 = N / 2

	// Allocate and populate CPU matrices
	hX := make([]float32, M*K)
	hW := make([]float32, K*N)
	hW1 := make([]float32, K*N1)
	hW2 := make([]float32, K*N2)

	r := rand.New(rand.NewSource(42))
	for i := range hX {
		hX[i] = (r.Float32() - 0.5) * 2.0
	}
	for rk := 0; rk < K; rk++ {
		for cn := 0; cn < N; cn++ {
			val := (r.Float32() - 0.5) * 2.0
			hW[rk*N+cn] = val
			if cn < N1 {
				hW1[rk*N1+cn] = val
			} else {
				hW2[rk*N2+(cn-N1)] = val
			}
		}
	}

	ctx0, err := devs[0].CreateContext()
	if err != nil {
		log.Fatalf("CreateContext 0 failed: %v", err)
	}
	defer ctx0.Destroy()

	mod0, err := ctx0.LoadModuleData(gemm.PTX)
	if err != nil {
		log.Fatalf("LoadModuleData 0 failed: %v", err)
	}
	defer mod0.Unload()

	fn0, err := mod0.Function("gemm")
	if err != nil {
		log.Fatalf("Function 0 failed: %v", err)
	}

	stream0, err := ctx0.CreateStream()
	if err != nil {
		log.Fatalf("CreateStream 0 failed: %v", err)
	}
	defer stream0.Destroy()

	// 1. Compute Full Baseline on GPU 0
	bytesX := uint64(M * K * 4)
	bytesW := uint64(K * N * 4)
	bytesY := uint64(M * N * 4)

	dX0, _ := ctx0.Alloc(bytesX)
	defer ctx0.Free(dX0)
	dW0, _ := ctx0.Alloc(bytesW)
	defer ctx0.Free(dW0)
	dY0, _ := ctx0.Alloc(bytesY)
	defer ctx0.Free(dY0)

	_ = ctx0.CopyHtoD(dX0, unsafe.Slice((*byte)(unsafe.Pointer(&hX[0])), bytesX))
	_ = ctx0.CopyHtoD(dW0, unsafe.Slice((*byte)(unsafe.Pointer(&hW[0])), bytesW))

	cfgFull := driver.LaunchConfig{
		GridDimX:  uint32((N + 15) / 16),
		GridDimY:  uint32((M + 15) / 16),
		BlockDimX: 16,
		BlockDimY: 16,
		Stream:    stream0,
	}

	_ = fn0.Launch(cfgFull, dX0, dW0, dY0, int32(M), int32(N), int32(K))
	_ = stream0.Synchronize()

	hYBaseline := make([]float32, M*N)
	_ = ctx0.CopyDtoH(unsafe.Slice((*byte)(unsafe.Pointer(&hYBaseline[0])), bytesY), dY0)

	// 2. Compute Column-Parallel Shards (TP=2)
	fmt.Printf("Sharding Projection Matrix W [%dx%d] -> 2 Shards of [%dx%d]\n", K, N, K, N1)

	var ctx1 *driver.Context
	var stream1 *driver.Stream
	var fn1 *driver.Function

	if numDevs >= 2 {
		ctx1, _ = devs[1].CreateContext()
		defer ctx1.Destroy()
		mod1, _ := ctx1.LoadModuleData(gemm.PTX)
		defer mod1.Unload()
		fn1, _ = mod1.Function("gemm")
		stream1, _ = ctx1.CreateStream()
		defer stream1.Destroy()
		_ = ctx0.EnablePeerAccess(ctx1)
		_ = ctx1.EnablePeerAccess(ctx0)
		fmt.Println("Topology: Multi-GPU Tensor Parallelism over Peer GPU contexts")
	} else {
		ctx1 = ctx0
		fn1 = fn0
		stream1, _ = ctx0.CreateStream()
		defer stream1.Destroy()
		fmt.Println("Topology: Single-GPU Tensor Parallelism over Concurrent Asynchronous Streams")
	}

	bytesW_shard := uint64(K * N1 * 4)
	bytesY_shard := uint64(M * N1 * 4)

	// Shard 1 buffers on ctx0
	dW_shard1, _ := ctx0.Alloc(bytesW_shard)
	defer ctx0.Free(dW_shard1)
	dY_shard1, _ := ctx0.Alloc(bytesY_shard)
	defer ctx0.Free(dY_shard1)
	_ = ctx0.CopyHtoD(dW_shard1, unsafe.Slice((*byte)(unsafe.Pointer(&hW1[0])), bytesW_shard))

	// Shard 2 buffers on ctx1
	var dX1 driver.DevicePtr
	if ctx1 != ctx0 {
		dX1, _ = ctx1.Alloc(bytesX)
		defer ctx1.Free(dX1)
		_ = ctx1.CopyHtoD(dX1, unsafe.Slice((*byte)(unsafe.Pointer(&hX[0])), bytesX))
	} else {
		dX1 = dX0
	}

	dW_shard2, _ := ctx1.Alloc(bytesW_shard)
	defer ctx1.Free(dW_shard2)
	dY_shard2, _ := ctx1.Alloc(bytesY_shard)
	defer ctx1.Free(dY_shard2)
	_ = ctx1.CopyHtoD(dW_shard2, unsafe.Slice((*byte)(unsafe.Pointer(&hW2[0])), bytesW_shard))

	cfgShard := driver.LaunchConfig{
		GridDimX:  uint32((N1 + 15) / 16),
		GridDimY:  uint32((M + 15) / 16),
		BlockDimX: 16,
		BlockDimY: 16,
	}

	start := time.Now()

	// Launch Shard 1: Y1 = X * W1
	cfgShard0 := cfgShard
	cfgShard0.Stream = stream0
	if err := fn0.Launch(cfgShard0, dX0, dW_shard1, dY_shard1, int32(M), int32(N1), int32(K)); err != nil {
		log.Fatalf("Launch shard 0 failed: %v", err)
	}

	// Launch Shard 2: Y2 = X * W2 (concurrently on stream 1)
	cfgShard1 := cfgShard
	cfgShard1.Stream = stream1
	if err := fn1.Launch(cfgShard1, dX1, dW_shard2, dY_shard2, int32(M), int32(N1), int32(K)); err != nil {
		log.Fatalf("Launch shard 1 failed: %v", err)
	}

	// Synchronize both shards
	_ = stream0.Synchronize()
	_ = stream1.Synchronize()
	elapsed := time.Since(start)

	// Copy back shard results
	hY1 := make([]float32, M*N1)
	hY2 := make([]float32, M*N2)
	_ = ctx0.CopyDtoH(unsafe.Slice((*byte)(unsafe.Pointer(&hY1[0])), bytesY_shard), dY_shard1)
	_ = ctx1.CopyDtoH(unsafe.Slice((*byte)(unsafe.Pointer(&hY2[0])), bytesY_shard), dY_shard2)

	// Concatenate Y = [Y1 | Y2] and compare against baseline
	maxDiff := float64(0.0)
	for rIdx := 0; rIdx < M; rIdx++ {
		for cIdx := 0; cIdx < N; cIdx++ {
			var shardVal float32
			if cIdx < N1 {
				shardVal = hY1[rIdx*N1+cIdx]
			} else {
				shardVal = hY2[rIdx*N2+(cIdx-N1)]
			}
			baseVal := hYBaseline[rIdx*N+cIdx]
			diff := math.Abs(float64(shardVal - baseVal))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
	}

	fmt.Printf("Parallel Sharded Execution Time: %v (%.2f ms)\n", elapsed, float64(elapsed.Microseconds())/1000.0)
	fmt.Printf("Max Absolute Difference vs Unsharded Baseline: %e\n", maxDiff)
	if maxDiff > 1e-4 {
		log.Fatalf("Verification FAILED: maxDiff %e > 1e-4", maxDiff)
	}
	fmt.Println("Verification: PASS (all column shards match baseline perfectly)")
	fmt.Println("================================================================================")
}
