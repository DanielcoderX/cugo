//go:build windows || linux

package main

import (
	"fmt"
	"log"
	"unsafe"

	"github.com/DanielcoderX/cugo/driver"
	"github.com/DanielcoderX/cugo/kernels/vecadd"
)

func float32SliceToBytes(s []float32) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*4)
}

func main() {
	fmt.Println("cugo: Asynchronous Streams & Events Example (no cgo)")
	fmt.Println("---------------------------------------------------")

	if err := driver.Init(); err != nil {
		log.Fatalf("driver.Init failed: %v", err)
	}

	devs, err := driver.Devices()
	if err != nil || len(devs) == 0 {
		log.Fatal("no CUDA devices found")
	}
	dev := devs[0]
	name, _ := dev.Name()
	fmt.Printf("Device: %s\n\n", name)

	ctx, err := dev.CreateContext()
	if err != nil {
		log.Fatalf("CreateContext failed: %v", err)
	}
	defer ctx.Destroy()

	// Load module
	mod, err := ctx.LoadModuleData(vecadd.PTX)
	if err != nil {
		log.Fatalf("LoadModuleData failed: %v", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("vecAdd")
	if err != nil {
		log.Fatalf("Function failed: %v", err)
	}

	// Create 2 independent streams
	stream1, err := ctx.CreateStream()
	if err != nil {
		log.Fatalf("CreateStream 1 failed: %v", err)
	}
	defer stream1.Destroy()

	stream2, err := ctx.CreateStream()
	if err != nil {
		log.Fatalf("CreateStream 2 failed: %v", err)
	}
	defer stream2.Destroy()

	// Create events for timing
	startEvent, err := ctx.CreateEvent()
	if err != nil {
		log.Fatalf("CreateEvent start failed: %v", err)
	}
	defer startEvent.Destroy()

	stopEvent, err := ctx.CreateEvent()
	if err != nil {
		log.Fatalf("CreateEvent stop failed: %v", err)
	}
	defer stopEvent.Destroy()

	// Workload: 2 chunks of 1M float32 elements
	const chunkSize = 1000000
	const byteSize = chunkSize * 4

	hA1 := make([]float32, chunkSize)
	hB1 := make([]float32, chunkSize)
	hA2 := make([]float32, chunkSize)
	hB2 := make([]float32, chunkSize)
	for i := 0; i < chunkSize; i++ {
		hA1[i] = float32(i)
		hB1[i] = 1.0
		hA2[i] = float32(i) * 2
		hB2[i] = 2.0
	}

	// Alloc GPU memory for both chunks
	dA1, _ := ctx.Alloc(byteSize); defer ctx.Free(dA1)
	dB1, _ := ctx.Alloc(byteSize); defer ctx.Free(dB1)
	dC1, _ := ctx.Alloc(byteSize); defer ctx.Free(dC1)

	dA2, _ := ctx.Alloc(byteSize); defer ctx.Free(dA2)
	dB2, _ := ctx.Alloc(byteSize); defer ctx.Free(dB2)
	dC2, _ := ctx.Alloc(byteSize); defer ctx.Free(dC2)

	cfg1 := driver.LaunchConfig{
		GridDimX:  (chunkSize + 255) / 256,
		BlockDimX: 256,
		Stream:    stream1,
	}
	cfg2 := driver.LaunchConfig{
		GridDimX:  (chunkSize + 255) / 256,
		BlockDimX: 256,
		Stream:    stream2,
	}

	fmt.Println("Pipelining Stream 1 & Stream 2 execution...")
	if err := startEvent.Record(stream1); err != nil {
		log.Fatalf("Record startEvent failed: %v", err)
	}

	// Stream 1 pipeline: HtoD -> Launch -> DtoH
	_ = stream1.CopyHtoDAsync(dA1, float32SliceToBytes(hA1))
	_ = stream1.CopyHtoDAsync(dB1, float32SliceToBytes(hB1))
	_ = fn.Launch(cfg1, dA1, dB1, dC1, int32(chunkSize))

	// Stream 2 pipeline: concurrent HtoD -> Launch -> DtoH
	_ = stream2.CopyHtoDAsync(dA2, float32SliceToBytes(hA2))
	_ = stream2.CopyHtoDAsync(dB2, float32SliceToBytes(hB2))
	_ = fn.Launch(cfg2, dA2, dB2, dC2, int32(chunkSize))

	res1 := make([]byte, byteSize)
	res2 := make([]byte, byteSize)
	_ = stream1.CopyDtoHAsync(res1, dC1)
	_ = stream2.CopyDtoHAsync(res2, dC2)

	// Synchronize both streams
	if err := stream1.Synchronize(); err != nil {
		log.Fatalf("stream1.Synchronize failed: %v", err)
	}
	if err := stream2.Synchronize(); err != nil {
		log.Fatalf("stream2.Synchronize failed: %v", err)
	}

	if err := stopEvent.Record(stream1); err != nil {
		log.Fatalf("Record stopEvent failed: %v", err)
	}
	_ = stopEvent.Synchronize()

	elapsedMs, err := driver.ElapsedTime(startEvent, stopEvent)
	if err != nil {
		log.Fatalf("driver.ElapsedTime failed: %v", err)
	}

	fmt.Printf("Completed concurrent processing of 2M elements across 2 streams.\n")
	fmt.Printf("GPU Elapsed Time: %.3f ms\n", elapsedMs)
	fmt.Printf("Throughput:       %.2f M elements/sec\n", float64(chunkSize*2)/(float64(elapsedMs)/1000.0)/1e6)
}
