//go:build windows || linux

package main

import (
	"fmt"
	"log"
	"math"
	"time"
	"unsafe"

	"github.com/DanielcoderX/cugo/driver"
	"github.com/DanielcoderX/cugo/kernels/vecadd"
)

func main() {
	fmt.Println("================================================================================")
	fmt.Println(" cugo: 3-Stage Overlapped Multi-Stream Pipeline (Zero CGO)")
	fmt.Println("================================================================================")

	if err := driver.Init(); err != nil {
		log.Fatalf("driver.Init failed: %v", err)
	}

	devs, err := driver.Devices()
	if err != nil || len(devs) == 0 {
		log.Fatal("no CUDA devices found")
	}
	dev := devs[0]
	devName, _ := dev.Name()
	fmt.Printf("Device: %s\n\n", devName)

	ctx, err := dev.CreateContext()
	if err != nil {
		log.Fatalf("CreateContext failed: %v", err)
	}
	defer ctx.Destroy()

	mod, err := ctx.LoadModuleData(vecadd.PTX)
	if err != nil {
		log.Fatalf("LoadModuleData failed: %v", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("vecAdd")
	if err != nil {
		log.Fatalf("Function failed: %v", err)
	}

	// 16,000,000 elements (64 MB per vector, 192 MB total memory traffic)
	const totalElements = 16 * 1024 * 1024
	const numChunks = 8
	const chunkSize = totalElements / numChunks
	const chunkBytes = uint64(chunkSize * 4)
	const totalBytes = uint64(totalElements * 4)

	// Allocate page-locked pinned host memory for high-speed concurrent DMA
	hA, err := ctx.AllocHost(totalBytes)
	if err != nil {
		log.Fatalf("AllocHost hA failed: %v", err)
	}
	defer hA.Free()

	hB, err := ctx.AllocHost(totalBytes)
	if err != nil {
		log.Fatalf("AllocHost hB failed: %v", err)
	}
	defer hB.Free()

	hC, err := ctx.AllocHost(totalBytes)
	if err != nil {
		log.Fatalf("AllocHost hC failed: %v", err)
	}
	defer hC.Free()


	// Initialize inputs
	sA := unsafe.Slice((*float32)(hA.Pointer()), totalElements)
	sB := unsafe.Slice((*float32)(hB.Pointer()), totalElements)
	sC := unsafe.Slice((*float32)(hC.Pointer()), totalElements)
	for i := 0; i < totalElements; i++ {
		sA[i] = float32(i % 100)
		sB[i] = float32(100 - (i % 100))
	}

	// Device buffers for each pipeline chunk
	type ChunkBuffers struct {
		dA, dB, dC driver.DevicePtr
	}
	buffers := make([]ChunkBuffers, numChunks)
	for i := 0; i < numChunks; i++ {
		dA, err := ctx.Alloc(chunkBytes)
		if err != nil {
			log.Fatalf("ctx.Alloc failed: %v", err)
		}
		defer ctx.Free(dA)
		dB, err := ctx.Alloc(chunkBytes)
		if err != nil {
			log.Fatalf("ctx.Alloc failed: %v", err)
		}
		defer ctx.Free(dB)
		dC, err := ctx.Alloc(chunkBytes)
		if err != nil {
			log.Fatalf("ctx.Alloc failed: %v", err)
		}
		defer ctx.Free(dC)
		buffers[i] = ChunkBuffers{dA: dA, dB: dB, dC: dC}
	}

	// Create 4 concurrent CUDA streams for interleaved pipeline
	const numStreams = 4
	streams := make([]*driver.Stream, numStreams)
	for s := 0; s < numStreams; s++ {
		st, err := ctx.CreateStream()
		if err != nil {
			log.Fatalf("CreateStream failed: %v", err)
		}
		defer st.Destroy()
		streams[s] = st
	}

	cfg := driver.LaunchConfig{
		GridDimX:  uint32((chunkSize + 255) / 256),
		BlockDimX: 256,
	}

	fmt.Printf("Total workload: %d elements (%.2f MB per vector, %d chunks)\n",
		totalElements, float64(totalBytes)/(1024*1024), numChunks)
	fmt.Printf("Executing 3-stage async pipeline across %d streams...\n", numStreams)

	// Pipeline execution:
	// For chunk i: HtoD Async -> Kernel Launch -> DtoH Async
	// Dispatched across circular streams to overlap H2D copy engine, SM compute, and D2H copy engine
	start := time.Now()

	for i := 0; i < numChunks; i++ {
		stream := streams[i%numStreams]
		cfg.Stream = stream

		offset := uintptr(i) * uintptr(chunkBytes)
		hAChunk := unsafe.Add(hA.Pointer(), offset)
		hBChunk := unsafe.Add(hB.Pointer(), offset)
		hCChunk := unsafe.Add(hC.Pointer(), offset)

		buf := buffers[i]

		// Stage 1: Async Host-to-Device transfer
		if err := stream.CopyHtoDAsync(buf.dA, unsafe.Slice((*byte)(hAChunk), chunkBytes)); err != nil {
			log.Fatalf("CopyHtoDAsync A failed: %v", err)
		}
		if err := stream.CopyHtoDAsync(buf.dB, unsafe.Slice((*byte)(hBChunk), chunkBytes)); err != nil {
			log.Fatalf("CopyHtoDAsync B failed: %v", err)
		}

		// Stage 2: Asynchronous Compute Kernel
		if err := fn.Launch(cfg, buf.dA, buf.dB, buf.dC, int32(chunkSize)); err != nil {
			log.Fatalf("fn.Launch failed: %v", err)
		}

		// Stage 3: Async Device-to-Host transfer
		if err := stream.CopyDtoHAsync(unsafe.Slice((*byte)(hCChunk), chunkBytes), buf.dC); err != nil {
			log.Fatalf("CopyDtoHAsync failed: %v", err)
		}
	}

	// Synchronize all streams
	for s := 0; s < numStreams; s++ {
		if err := streams[s].Synchronize(); err != nil {
			log.Fatalf("Stream synchronize failed: %v", err)
		}
	}
	elapsed := time.Since(start)

	// Verify correctness
	for i := 0; i < totalElements; i++ {
		if math.Abs(float64(sC[i]-100.0)) > 1e-4 {
			log.Fatalf("Verification failed at element %d: got %f, expected 100.0", i, sC[i])
		}
	}

	fmt.Printf("Pipelined Execution Time: %v (%.2f ms)\n", elapsed, float64(elapsed.Microseconds())/1000.0)
	totalTrafficGB := float64(totalBytes*3) / 1e9 // 2 inputs read + 1 output written
	effectiveBandwidth := totalTrafficGB / elapsed.Seconds()
	fmt.Printf("Effective End-to-End Throughput: %.2f GB/s\n", effectiveBandwidth)
	fmt.Println("Verification: PASS (16M elements match 100.0)")
	fmt.Println("================================================================================")
}
