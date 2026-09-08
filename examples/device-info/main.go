//go:build windows

package main

import (
	"fmt"
	"log"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/internal/nvapi"
)

func main() {
	fmt.Println("cugo: pure-Go CUDA Driver API (no cgo)")
	fmt.Println("--------------------------------------")

	if err := driver.Init(); err != nil {
		log.Fatalf("driver.Init failed: %v", err)
	}

	driverVersion, err := driver.DriverVersion()
	if err != nil {
		log.Fatalf("driver.DriverVersion failed: %v", err)
	}
	fmt.Printf("CUDA Driver Version: %d.%d (raw %d)\n\n",
		driverVersion/1000, (driverVersion%100)/10, driverVersion)

	devs, err := driver.Devices()
	if err != nil {
		log.Fatalf("driver.Devices failed: %v", err)
	}

	fmt.Printf("Detected %d CUDA-capable device(s):\n\n", len(devs))
	for i, dev := range devs {
		name, err := dev.Name()
		if err != nil {
			log.Fatalf("dev[%d].Name failed: %v", i, err)
		}

		major, minor, err := dev.ComputeCapability()
		if err != nil {
			log.Fatalf("dev[%d].ComputeCapability failed: %v", i, err)
		}

		totalMem, err := dev.TotalMemory()
		if err != nil {
			log.Fatalf("dev[%d].TotalMemory failed: %v", i, err)
		}

		smCount, _ := dev.Attribute(nvapi.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
		maxThreadsPerBlock, _ := dev.Attribute(nvapi.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
		clockRateKHz, _ := dev.Attribute(nvapi.CU_DEVICE_ATTRIBUTE_CLOCK_RATE)
		memClockKHz, _ := dev.Attribute(nvapi.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
		busWidth, _ := dev.Attribute(nvapi.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
		l2CacheBytes, _ := dev.Attribute(nvapi.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)

		fmt.Printf("[%d] %s\n", i, name)
		fmt.Printf("    Compute Capability:        sm_%d%d (%d.%d)\n", major, minor, major, minor)
		fmt.Printf("    Total Global Memory:       %.2f GiB (%d bytes)\n", float64(totalMem)/(1024*1024*1024), totalMem)
		fmt.Printf("    Multiprocessors (SMs):     %d\n", smCount)
		fmt.Printf("    Max Threads Per Block:     %d\n", maxThreadsPerBlock)
		fmt.Printf("    GPU Base/Boost Clock:      %.2f MHz\n", float64(clockRateKHz)/1000.0)
		fmt.Printf("    Memory Clock Rate:         %.2f MHz\n", float64(memClockKHz)/1000.0)
		fmt.Printf("    Memory Bus Width:          %d-bit\n", busWidth)
		fmt.Printf("    L2 Cache Size:             %d KB\n", l2CacheBytes/1024)
		fmt.Println()
	}
}
