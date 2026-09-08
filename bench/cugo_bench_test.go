//go:build windows

package bench_test

import (
	"runtime"
	"testing"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/internal/nvapi"
	"github.com/cugo/cugo/kernels/vecadd"
)

func BenchmarkDriverCallOverhead(b *testing.B) {
	if err := nvapi.CheckDriver(); err != nil {
		b.Skip("no driver")
	}
	_ = driver.Init()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var count int32
		_ = nvapi.CuDeviceGetCount(&count)
	}
}

func BenchmarkKernelLaunchLatency(b *testing.B) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := nvapi.CheckDriver(); err != nil {
		b.Skip("no driver")
	}
	if err := driver.Init(); err != nil {
		b.Fatal(err)
	}
	devs, err := driver.Devices()
	if err != nil || len(devs) == 0 {
		b.Skip("no devices")
	}

	ctx, err := devs[0].CreateContext()
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.Destroy()

	mod, err := ctx.LoadModuleData(vecadd.PTX)
	if err != nil {
		b.Fatal(err)
	}
	defer mod.Unload()

	fn, err := mod.Function("vecAdd")
	if err != nil {
		b.Fatal(err)
	}

	// 1-element dummy buffer
	dptr, _ := ctx.Alloc(4)
	defer ctx.Free(dptr)

	cfg := driver.LaunchConfig{
		GridDimX:  1,
		BlockDimX: 1,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = fn.Launch(cfg, dptr, dptr, dptr, int32(1))
	}
}

func benchmarkMemoryBandwidth(b *testing.B, sizeBytes int, direction string) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := nvapi.CheckDriver(); err != nil {
		b.Skip("no driver")
	}
	if err := driver.Init(); err != nil {
		b.Fatal(err)
	}
	devs, err := driver.Devices()
	if err != nil || len(devs) == 0 {
		b.Skip("no devices")
	}

	ctx, err := devs[0].CreateContext()
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.Destroy()

	dptr, err := ctx.Alloc(uint64(sizeBytes))
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.Free(dptr)

	hostBuf := make([]byte, sizeBytes)

	b.SetBytes(int64(sizeBytes))
	b.ResetTimer()
	b.ReportAllocs()

	if direction == "HtoD" {
		for i := 0; i < b.N; i++ {
			if err := ctx.CopyHtoD(dptr, hostBuf); err != nil {
				b.Fatal(err)
			}
		}
	} else {
		for i := 0; i < b.N; i++ {
			if err := ctx.CopyDtoH(hostBuf, dptr); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkMemcpyHtoD_1MB(b *testing.B)  { benchmarkMemoryBandwidth(b, 1024*1024, "HtoD") }
func BenchmarkMemcpyHtoD_16MB(b *testing.B) { benchmarkMemoryBandwidth(b, 16*1024*1024, "HtoD") }
func BenchmarkMemcpyDtoH_1MB(b *testing.B)  { benchmarkMemoryBandwidth(b, 1024*1024, "DtoH") }
func BenchmarkMemcpyDtoH_16MB(b *testing.B) { benchmarkMemoryBandwidth(b, 16*1024*1024, "DtoH") }

func benchmarkPinnedMemoryBandwidth(b *testing.B, sizeBytes int, direction string) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := nvapi.CheckDriver(); err != nil {
		b.Skip("no driver")
	}
	if err := driver.Init(); err != nil {
		b.Fatal(err)
	}
	devs, err := driver.Devices()
	if err != nil || len(devs) == 0 {
		b.Skip("no devices")
	}

	ctx, err := devs[0].CreateContext()
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.Destroy()

	dptr, err := ctx.Alloc(uint64(sizeBytes))
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.Free(dptr)

	hMem, err := ctx.AllocHost(uint64(sizeBytes))
	if err != nil {
		b.Fatal(err)
	}
	defer hMem.Free()

	stream, err := ctx.CreateStream()
	if err != nil {
		b.Fatal(err)
	}
	defer stream.Destroy()

	hostBuf := hMem.Bytes()

	b.SetBytes(int64(sizeBytes))
	b.ResetTimer()
	b.ReportAllocs()

	if direction == "HtoD" {
		for i := 0; i < b.N; i++ {
			if err := stream.CopyHtoDAsync(dptr, hostBuf); err != nil {
				b.Fatal(err)
			}
			if err := stream.Synchronize(); err != nil {
				b.Fatal(err)
			}
		}
	} else {
		for i := 0; i < b.N; i++ {
			if err := stream.CopyDtoHAsync(hostBuf, dptr); err != nil {
				b.Fatal(err)
			}
			if err := stream.Synchronize(); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkPinnedMemcpyHtoD_16MB(b *testing.B) {
	benchmarkPinnedMemoryBandwidth(b, 16*1024*1024, "HtoD")
}

func BenchmarkPinnedMemcpyDtoH_16MB(b *testing.B) {
	benchmarkPinnedMemoryBandwidth(b, 16*1024*1024, "DtoH")
}
