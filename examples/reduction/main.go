//go:build windows || linux

package main

import (
	"fmt"
	"log"
	"runtime"
	"time"
	"unsafe"

	"github.com/DanielcoderX/cugo/driver"
	"github.com/DanielcoderX/cugo/kernels/reduction"
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
	fmt.Printf("Running Warp-Shuffle Parallel Reduction on %s\n", name)

	ctx, err := dev.CreateContext()
	if err != nil {
		log.Fatalf("CreateContext failed: %v", err)
	}
	defer ctx.Destroy()

	mod, err := ctx.LoadModuleData(reduction.PTX)
	if err != nil {
		log.Fatalf("LoadModuleData failed: %v", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("reduce_sum")
	if err != nil {
		log.Fatalf("Function reduce_sum failed: %v", err)
	}

	const n = 10_000_000 // 10 million elements
	const bytes = n * 4

	hData := make([]float32, n)
	for i := range hData {
		hData[i] = 1.0
	}

	dIn, err := ctx.Alloc(bytes); if err != nil { log.Fatal(err) }
	defer ctx.Free(dIn)
	dOut, err := ctx.Alloc(4); if err != nil { log.Fatal(err) }
	defer ctx.Free(dOut)

	rawIn := unsafe.Slice((*byte)(unsafe.Pointer(&hData[0])), bytes)
	if err := ctx.CopyHtoD(dIn, rawIn); err != nil { log.Fatal(err) }

	zero := float32(0.0)
	rawZero := unsafe.Slice((*byte)(unsafe.Pointer(&zero)), 4)
	if err := ctx.CopyHtoD(dOut, rawZero); err != nil { log.Fatal(err) }

	cfg := driver.LaunchConfig{
		GridDimX:  128,
		BlockDimX: 256,
	}

	// Warmup
	_ = fn.Launch(cfg, dIn, dOut, int32(n))
	_ = ctx.Synchronize()

	// Benchmark reduction over 50 iterations
	const iters = 50
	_ = ctx.CopyHtoD(dOut, rawZero)
	start := time.Now()
	for i := 0; i < iters; i++ {
		if err := fn.Launch(cfg, dIn, dOut, int32(n)); err != nil {
			log.Fatal(err)
		}
	}
	if err := ctx.Synchronize(); err != nil {
		log.Fatal(err)
	}
	elapsed := time.Since(start)
	avgLatency := elapsed / iters

	var result float32
	if err := ctx.CopyDtoH(unsafe.Slice((*byte)(unsafe.Pointer(&result)), 4), dOut); err != nil {
		log.Fatal(err)
	}

	bandwidth := (float64(bytes) / avgLatency.Seconds()) / 1e9
	fmt.Printf("Elements: %d (%.2f MB)\n", n, float64(bytes)/(1024*1024))
	fmt.Printf("Avg Kernel Latency: %v\n", avgLatency)
	fmt.Printf("Effective Memory Bandwidth: %.2f GB/s\n", bandwidth)
	fmt.Printf("Reduction Result: %.2f (expected %.2f)\n", result/float32(iters), float32(n))
}
