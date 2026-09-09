//go:build windows

package bench_test

import (
	"runtime"
	"testing"
	"unsafe"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/internal/nvapi"
	"github.com/cugo/cugo/kernels/vecadd"
)

func BenchmarkDriverCallOverhead(b *testing.B) {
	if err := nvapi.CheckDriver(); err != nil {
		b.Skip("no driver")
	}
	if err := driver.Init(); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var count int32
		_ = nvapi.CuDeviceGetCount(&count)
	}
}

func BenchmarkVecAddRoundTrip_100K(b *testing.B) {
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

	const n = 100000
	const byteSize = n * 4

	hA := make([]float32, n)
	hB := make([]float32, n)
	hC := make([]float32, n)
	for i := 0; i < n; i++ {
		hA[i] = float32(i)
		hB[i] = float32(i * 2)
	}

	dA, err := ctx.Alloc(byteSize)
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.Free(dA)

	dB, err := ctx.Alloc(byteSize)
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.Free(dB)

	dC, err := ctx.Alloc(byteSize)
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.Free(dC)

	rawA := unsafe.Slice((*byte)(unsafe.Pointer(&hA[0])), byteSize)
	rawB := unsafe.Slice((*byte)(unsafe.Pointer(&hB[0])), byteSize)
	rawC := unsafe.Slice((*byte)(unsafe.Pointer(&hC[0])), byteSize)

	cfg := driver.LaunchConfig{
		GridDimX:  (n + 255) / 256,
		BlockDimX: 256,
	}

	b.SetBytes(byteSize * 3) // 2 reads + 1 write
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		if err := ctx.CopyHtoD(dA, rawA); err != nil {
			b.Fatal(err)
		}
		if err := ctx.CopyHtoD(dB, rawB); err != nil {
			b.Fatal(err)
		}
		if err := fn.Launch(cfg, dA, dB, dC, int32(n)); err != nil {
			b.Fatal(err)
		}
		if err := ctx.CopyDtoH(rawC, dC); err != nil {
			b.Fatal(err)
		}
	}
}
