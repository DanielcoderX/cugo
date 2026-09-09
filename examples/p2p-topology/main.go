//go:build windows || linux

package main

import (
	"fmt"
	"log"
	"time"

	"github.com/DanielcoderX/cugo/driver"
	"github.com/DanielcoderX/cugo/internal/nvapi"
)

func main() {
	fmt.Println("================================================================================")
	fmt.Println(" cugo: Multi-GPU NVLink / PCIe P2P Topology Explorer & Benchmark")
	fmt.Println("================================================================================")

	if err := driver.Init(); err != nil {
		log.Fatalf("driver.Init failed: %v", err)
	}

	devs, err := driver.Devices()
	if err != nil {
		log.Fatalf("driver.Devices failed: %v", err)
	}

	numDevs := len(devs)
	fmt.Printf("Detected %d CUDA device(s)\n\n", numDevs)

	for i, dev := range devs {
		name, _ := dev.Name()
		ccMajor, ccMinor, _ := dev.ComputeCapability()
		totalMem, _ := dev.TotalMemory()
		fmt.Printf("  [GPU %d] %s (CC %d.%d, %.2f GiB)\n",
			i, name, ccMajor, ccMinor, float64(totalMem)/(1024*1024*1024))
	}
	fmt.Println()

	if numDevs < 2 {
		fmt.Println("P2P Matrix:")
		fmt.Println("  GPU 0 -> GPU 0: Self (Loopback, Direct Global Memory)")
		fmt.Println("\nNote: Single-GPU host detected. Multi-GPU P2P / NVLink transfers require >= 2 GPUs.")
		fmt.Println("================================================================================")
		return
	}

	fmt.Println("P2P Accessibility Matrix (CanAccessPeer):")
	fmt.Print("     ")
	for j := 0; j < numDevs; j++ {
		fmt.Printf(" GPU%d ", j)
	}
	fmt.Println()

	for i := 0; i < numDevs; i++ {
		fmt.Printf("GPU%d ", i)
		for j := 0; j < numDevs; j++ {
			if i == j {
				fmt.Print(" Self ")
				continue
			}
			canAccess, err := devs[i].CanAccessPeer(devs[j])
			if err != nil || !canAccess {
				fmt.Print("  No  ")
			} else {
				fmt.Print("  Yes ")
			}
		}
		fmt.Println()
	}
	fmt.Println()

	// Benchmark P2P transfers between accessible pairs
	const testBytes = uint64(64 * 1024 * 1024) // 64 MB
	for i := 0; i < numDevs; i++ {
		for j := 0; j < numDevs; j++ {
			if i == j {
				continue
			}
			canAccess, err := devs[i].CanAccessPeer(devs[j])
			if err != nil || !canAccess {
				continue
			}

			// Query P2P attributes
			rank, _ := devs[i].P2PAttribute(nvapi.CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK, devs[j])
			atomics, _ := devs[i].P2PAttribute(nvapi.CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED, devs[j])

			fmt.Printf("Testing GPU %d -> GPU %d (PerfRank: %d, NativeAtomics: %d)...\n",
				i, j, rank, atomics)

			ctxSrc, err := devs[i].CreateContext()
			if err != nil {
				continue
			}
			ctxDst, err := devs[j].CreateContext()
			if err != nil {
				ctxSrc.Destroy()
				continue
			}

			// Enable P2P access
			_ = ctxSrc.EnablePeerAccess(ctxDst)
			_ = ctxDst.EnablePeerAccess(ctxSrc)

			dSrc, errSrc := ctxSrc.Alloc(testBytes)
			dDst, errDst := ctxDst.Alloc(testBytes)

			if errSrc == nil && errDst == nil {
				// Warmup
				_ = driver.CopyPeer(ctxDst, dDst, ctxSrc, dSrc, testBytes)

				start := time.Now()
				const iters = 10
				for it := 0; it < iters; it++ {
					_ = driver.CopyPeer(ctxDst, dDst, ctxSrc, dSrc, testBytes)
				}
				elapsed := time.Since(start)
				sec := elapsed.Seconds() / float64(iters)
				gbPerSec := (float64(testBytes) / 1e9) / sec
				fmt.Printf("  GPU %d -> GPU %d P2P Bandwidth: %.2f GB/s\n", i, j, gbPerSec)
			}


			if dSrc != 0 {
				ctxSrc.Free(dSrc)
			}
			if dDst != 0 {
				ctxDst.Free(dDst)
			}
			_ = ctxSrc.DisablePeerAccess(ctxDst)
			_ = ctxDst.DisablePeerAccess(ctxSrc)
			ctxSrc.Destroy()
			ctxDst.Destroy()
		}
	}
	fmt.Println("================================================================================")
}
