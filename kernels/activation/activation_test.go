package activation_test

import (
	"math"
	"runtime"
	"testing"
	"unsafe"

	"github.com/DanielcoderX/cugo/driver"
	"github.com/DanielcoderX/cugo/internal/nvapi"
	"github.com/DanielcoderX/cugo/kernels/activation"
)

func cpuGELU(x float32) float32 {
	inner := 0.7978845608 * (float64(x) + 0.044715*float64(x)*float64(x)*float64(x))
	cdf := 0.5 * (1.0 + math.Tanh(inner))
	return float32(float64(x) * cdf)
}

func cpuSiLU(x float32) float32 {
	return x / (1.0 + float32(math.Exp(float64(-x))))
}

func cpuSigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

func TestActivationKernels(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := nvapi.CheckDriver(); err != nil {
		t.Skip("no CUDA driver")
	}
	if err := driver.Init(); err != nil {
		t.Skip("driver.Init failed")
	}
	devs, err := driver.Devices()
	if err != nil || len(devs) == 0 {
		t.Skip("no devices")
	}

	ctx, err := devs[0].CreateContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Destroy()

	mod, err := ctx.LoadModuleData(activation.PTX)
	if err != nil {
		t.Fatalf("LoadModuleData failed: %v", err)
	}
	defer mod.Unload()

	const n = 10000
	const byteSize = n * 4

	hIn := make([]float32, n)
	for i := range hIn {
		// Span from -5.0 to +5.0
		hIn[i] = -5.0 + float32(i)*10.0/float32(n)
	}

	dIn, err := ctx.Alloc(byteSize)
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Free(dIn)

	dOut, err := ctx.Alloc(byteSize)
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Free(dOut)

	inBytes := unsafe.Slice((*byte)(unsafe.Pointer(&hIn[0])), byteSize)
	if err := ctx.CopyHtoD(dIn, inBytes); err != nil {
		t.Fatal(err)
	}

	cfg := driver.LaunchConfig{
		GridDimX:  (n + 255) / 256,
		BlockDimX: 256,
	}

	hOut := make([]float32, n)
	outBytes := unsafe.Slice((*byte)(unsafe.Pointer(&hOut[0])), byteSize)

	// 1. GELU
	geluFn, err := mod.Function("geluKernel")
	if err != nil {
		t.Fatal(err)
	}
	if err := geluFn.Launch(cfg, dIn, dOut, int32(n)); err != nil {
		t.Fatalf("gelu launch failed: %v", err)
	}
	if err := ctx.CopyDtoH(outBytes, dOut); err != nil {
		t.Fatal(err)
	}
	for i, x := range hIn {
		expected := cpuGELU(x)
		if math.Abs(float64(hOut[i]-expected)) > 1e-3 {
			t.Fatalf("GELU mismatch at index %d: got %f, expected %f", i, hOut[i], expected)
		}
	}
	t.Logf("GELU validated across %d elements!", n)

	// 2. ReLU
	reluFn, err := mod.Function("reluKernel")
	if err != nil {
		t.Fatal(err)
	}
	if err := reluFn.Launch(cfg, dIn, dOut, int32(n)); err != nil {
		t.Fatalf("relu launch failed: %v", err)
	}
	if err := ctx.CopyDtoH(outBytes, dOut); err != nil {
		t.Fatal(err)
	}
	for i, x := range hIn {
		var expected float32
		if x > 0 {
			expected = x
		}
		if math.Abs(float64(hOut[i]-expected)) > 1e-4 {
			t.Fatalf("ReLU mismatch at index %d: got %f, expected %f", i, hOut[i], expected)
		}
	}
	t.Logf("ReLU validated across %d elements!", n)

	// 3. SiLU (Swish)
	siluFn, err := mod.Function("siluKernel")
	if err != nil {
		t.Fatal(err)
	}
	if err := siluFn.Launch(cfg, dIn, dOut, int32(n)); err != nil {
		t.Fatalf("silu launch failed: %v", err)
	}
	if err := ctx.CopyDtoH(outBytes, dOut); err != nil {
		t.Fatal(err)
	}
	for i, x := range hIn {
		expected := cpuSiLU(x)
		if math.Abs(float64(hOut[i]-expected)) > 1e-3 {
			t.Fatalf("SiLU mismatch at index %d: got %f, expected %f", i, hOut[i], expected)
		}
	}
	t.Logf("SiLU validated across %d elements!", n)

	// 4. Sigmoid
	sigFn, err := mod.Function("sigmoidKernel")
	if err != nil {
		t.Fatal(err)
	}
	if err := sigFn.Launch(cfg, dIn, dOut, int32(n)); err != nil {
		t.Fatalf("sigmoid launch failed: %v", err)
	}
	if err := ctx.CopyDtoH(outBytes, dOut); err != nil {
		t.Fatal(err)
	}
	for i, x := range hIn {
		expected := cpuSigmoid(x)
		if math.Abs(float64(hOut[i]-expected)) > 1e-3 {
			t.Fatalf("Sigmoid mismatch at index %d: got %f, expected %f", i, hOut[i], expected)
		}
	}
	t.Logf("Sigmoid validated across %d elements!", n)
}

func TestBiasAddKernel(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := nvapi.CheckDriver(); err != nil {
		t.Skip("no CUDA driver")
	}
	if err := driver.Init(); err != nil {
		t.Skip("driver.Init failed")
	}
	devs, err := driver.Devices()
	if err != nil || len(devs) == 0 {
		t.Skip("no devices")
	}

	ctx, err := devs[0].CreateContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Destroy()

	mod, err := ctx.LoadModuleData(activation.PTX)
	if err != nil {
		t.Fatalf("LoadModuleData failed: %v", err)
	}
	defer mod.Unload()

	biasFn, err := mod.Function("biasAddKernel")
	if err != nil {
		t.Fatal(err)
	}

	const rows = 32
	const cols = 64
	const totalElements = rows * cols

	hIn := make([]float32, totalElements)
	hBias := make([]float32, cols)
	for i := range hIn {
		hIn[i] = float32(i) * 0.1
	}
	for c := range hBias {
		hBias[c] = float32(c) * 2.0
	}

	dIn, err := ctx.Alloc(totalElements * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Free(dIn)

	dBias, err := ctx.Alloc(cols * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Free(dBias)

	dOut, err := ctx.Alloc(totalElements * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Free(dOut)

	if err := ctx.CopyHtoD(dIn, unsafe.Slice((*byte)(unsafe.Pointer(&hIn[0])), totalElements*4)); err != nil {
		t.Fatal(err)
	}
	if err := ctx.CopyHtoD(dBias, unsafe.Slice((*byte)(unsafe.Pointer(&hBias[0])), cols*4)); err != nil {
		t.Fatal(err)
	}

	cfg := driver.LaunchConfig{
		GridDimX:  (cols + 15) / 16,
		GridDimY:  (rows + 15) / 16,
		BlockDimX: 16,
		BlockDimY: 16,
	}

	if err := biasFn.Launch(cfg, dIn, dBias, dOut, int32(rows), int32(cols)); err != nil {
		t.Fatalf("biasAdd launch failed: %v", err)
	}

	hOut := make([]float32, totalElements)
	if err := ctx.CopyDtoH(unsafe.Slice((*byte)(unsafe.Pointer(&hOut[0])), totalElements*4), dOut); err != nil {
		t.Fatal(err)
	}

	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			idx := r*cols + c
			expected := hIn[idx] + hBias[c]
			if math.Abs(float64(hOut[idx]-expected)) > 1e-4 {
				t.Fatalf("bias mismatch at [%d, %d]: got %f, expected %f", r, c, hOut[idx], expected)
			}
		}
	}
	t.Logf("BiasAdd validated across %dx%d matrix on GPU!", rows, cols)
}
