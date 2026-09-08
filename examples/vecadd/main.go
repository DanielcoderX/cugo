//go:build windows || linux

package main

import (
	"fmt"
	"log"
	"math"
	"unsafe"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/kernels/vecadd"
)

func float32SliceToBytes(s []float32) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*4)
}

func bytesToFloat32Slice(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

func main() {
	fmt.Println("cugo: Vector Addition Kernel Example (no cgo)")
	fmt.Println("----------------------------------------------")

	// 1. Initialize CUDA Driver
	if err := driver.Init(); err != nil {
		log.Fatalf("driver.Init failed: %v", err)
	}

	devs, err := driver.Devices()
	if err != nil || len(devs) == 0 {
		log.Fatal("no CUDA-capable devices found")
	}
	dev := devs[0]
	name, _ := dev.Name()
	fmt.Printf("Using Device: %s\n\n", name)

	// 2. Create Context
	ctx, err := dev.CreateContext()
	if err != nil {
		log.Fatalf("CreateContext failed: %v", err)
	}
	defer ctx.Destroy()

	// 3. Load PTX Module and lookup vecAdd kernel
	mod, err := ctx.LoadModuleData(vecadd.PTX)
	if err != nil {
		log.Fatalf("LoadModuleData failed: %v", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("vecAdd")
	if err != nil {
		log.Fatalf("Function('vecAdd') failed: %v", err)
	}

	// 4. Prepare host test data
	const n = 100000
	const byteSize = n * 4

	hA := make([]float32, n)
	hB := make([]float32, n)
	for i := 0; i < n; i++ {
		hA[i] = float32(i) * 1.5
		hB[i] = float32(i) * 2.5
	}

	// 5. Allocate GPU memory
	dA, err := ctx.Alloc(byteSize)
	if err != nil {
		log.Fatalf("Alloc(dA) failed: %v", err)
	}
	defer ctx.Free(dA)

	dB, err := ctx.Alloc(byteSize)
	if err != nil {
		log.Fatalf("Alloc(dB) failed: %v", err)
	}
	defer ctx.Free(dB)

	dC, err := ctx.Alloc(byteSize)
	if err != nil {
		log.Fatalf("Alloc(dC) failed: %v", err)
	}
	defer ctx.Free(dC)

	// 6. Copy input data to GPU (HtoD)
	if err := ctx.CopyHtoD(dA, float32SliceToBytes(hA)); err != nil {
		log.Fatalf("CopyHtoD(dA) failed: %v", err)
	}
	if err := ctx.CopyHtoD(dB, float32SliceToBytes(hB)); err != nil {
		log.Fatalf("CopyHtoD(dB) failed: %v", err)
	}

	// 7. Configure grid and launch kernel
	const blockSize = 256
	gridSize := (uint32(n) + blockSize - 1) / blockSize

	cfg := driver.LaunchConfig{
		GridDimX:  gridSize,
		BlockDimX: blockSize,
	}

	fmt.Printf("Launching vecAdd kernel:\n")
	fmt.Printf("  Elements:   %d\n", n)
	fmt.Printf("  GridDim:    (%d, 1, 1)\n", gridSize)
	fmt.Printf("  BlockDim:   (%d, 1, 1)\n\n", blockSize)

	if err := fn.Launch(cfg, dA, dB, dC, int32(n)); err != nil {
		log.Fatalf("fn.Launch failed: %v", err)
	}

	// 8. Copy result back to host (DtoH)
	cBytes := make([]byte, byteSize)
	if err := ctx.CopyDtoH(cBytes, dC); err != nil {
		log.Fatalf("CopyDtoH(dC) failed: %v", err)
	}
	hC := bytesToFloat32Slice(cBytes)

	// 9. Verify results
	mismatches := 0
	for i := 0; i < n; i++ {
		expected := hA[i] + hB[i]
		if math.Abs(float64(hC[i]-expected)) > 1e-4 {
			if mismatches < 5 {
				fmt.Printf("Mismatch at index %d: got %f, expected %f\n", i, hC[i], expected)
			}
			mismatches++
		}
	}

	if mismatches == 0 {
		fmt.Printf("SUCCESS: All %d elements computed and verified correctly!\n", n)
		fmt.Printf("Sample check: hA[42]=%.1f + hB[42]=%.1f == hC[42]=%.1f\n",
			hA[42], hB[42], hC[42])
	} else {
		log.Fatalf("FAILED: %d mismatches found", mismatches)
	}
}
