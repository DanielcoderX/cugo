package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/kernels/activation"
	"github.com/cugo/cugo/kernels/gemm"
	"github.com/cugo/cugo/kernels/layernorm"
	"github.com/cugo/cugo/tensor"
)

func cpuMatMul(A []float32, B []float32, M, N, K int) []float32 {
	C := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += A[i*K+k] * B[k*N+j]
			}
			C[i*N+j] = sum
		}
	}
	return C
}

func cpuBiasAdd(X []float32, bias []float32, M, N int) {
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			X[i*N+j] += bias[j]
		}
	}
}

func cpuGELU(X []float32) {
	for i, x := range X {
		inner := 0.7978845608 * (float64(x) + 0.044715*float64(x)*float64(x)*float64(x))
		cdf := 0.5 * (1.0 + math.Tanh(inner))
		X[i] = float32(float64(x) * cdf)
	}
}

func main() {
	fmt.Println("================================================================================")
	fmt.Println(" cugo: Neural Network MLP Forward Pass (PyTorch-like Engine in Pure Go)")
	fmt.Println(" Architecture: X [16, 128] -> Linear(128->256) -> GELU -> Linear(256->64) -> RMSNorm")
	fmt.Println("================================================================================")

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

	// Load PTX modules
	gemmMod, err := ctx.LoadModuleData(gemm.PTX); if err != nil { log.Fatal(err) }
	defer gemmMod.Unload()
	gemmFn, _ := gemmMod.Function("gemm")

	actMod, err := ctx.LoadModuleData(activation.PTX); if err != nil { log.Fatal(err) }
	defer actMod.Unload()
	geluFn, _ := actMod.Function("geluKernel")
	biasFn, _ := actMod.Function("biasAddKernel")



	// Dimensions
	const batch = 16
	const dIn = 128
	const dHidden = 256
	const dOut = 64

	rng := rand.New(rand.NewSource(1337))

	// Host weights and inputs
	hX := make([]float32, batch*dIn)
	hW1 := make([]float32, dIn*dHidden)
	hb1 := make([]float32, dHidden)
	hW2 := make([]float32, dHidden*dOut)
	hb2 := make([]float32, dOut)
	hWeightNorm := make([]float32, dOut)

	for i := range hX { hX[i] = rng.Float32()*2.0 - 1.0 }
	for i := range hW1 { hW1[i] = (rng.Float32()*2.0 - 1.0) / float32(math.Sqrt(float64(dIn))) }
	for i := range hb1 { hb1[i] = 0.01 }
	for i := range hW2 { hW2[i] = (rng.Float32()*2.0 - 1.0) / float32(math.Sqrt(float64(dHidden))) }
	for i := range hb2 { hb2[i] = 0.01 }
	for i := range hWeightNorm { hWeightNorm[i] = 1.0 }

	// Create GPU Tensors
	tX, _ := tensor.NewFromFloat32(ctx, hX, batch, dIn); defer tX.Close()
	tW1, _ := tensor.NewFromFloat32(ctx, hW1, dIn, dHidden); defer tW1.Close()
	tb1, _ := tensor.NewFromFloat32(ctx, hb1, dHidden); defer tb1.Close()
	tW2, _ := tensor.NewFromFloat32(ctx, hW2, dHidden, dOut); defer tW2.Close()
	tb2, _ := tensor.NewFromFloat32(ctx, hb2, dOut); defer tb2.Close()
	tWeightNorm, _ := tensor.NewFromFloat32(ctx, hWeightNorm, dOut); defer tWeightNorm.Close()

	// Intermediates
	tH, _ := tensor.New(ctx, tensor.Float32, batch, dHidden); defer tH.Close()
	tY, _ := tensor.New(ctx, tensor.Float32, batch, dOut); defer tY.Close()
	tOut, _ := tensor.New(ctx, tensor.Float32, batch, dOut); defer tOut.Close()

	start := time.Now()

	// 1. Layer 1 GEMM: tH = tX * tW1
	cfgGEMM1 := driver.LaunchConfig{
		GridDimX:  uint32((dHidden + 15) / 16),
		GridDimY:  uint32((batch + 15) / 16),
		BlockDimX: 16,
		BlockDimY: 16,
	}
	if err := gemmFn.Launch(cfgGEMM1, tX.DevicePtr(), tW1.DevicePtr(), tH.DevicePtr(), int32(batch), int32(dHidden), int32(dIn)); err != nil {
		log.Fatalf("gemm1 failed: %v", err)
	}

	// 2. Bias add: tH += tb1
	cfgBias1 := driver.LaunchConfig{
		GridDimX:  uint32((dHidden + 15) / 16),
		GridDimY:  uint32((batch + 15) / 16),
		BlockDimX: 16,
		BlockDimY: 16,
	}
	if err := biasFn.Launch(cfgBias1, tH.DevicePtr(), tb1.DevicePtr(), tH.DevicePtr(), int32(batch), int32(dHidden)); err != nil {
		log.Fatalf("bias1 failed: %v", err)
	}

	// 3. Activation: tH = GELU(tH)
	const numH = batch * dHidden
	cfgAct := driver.LaunchConfig{
		GridDimX:  uint32((numH + 255) / 256),
		BlockDimX: 256,
	}
	if err := geluFn.Launch(cfgAct, tH.DevicePtr(), tH.DevicePtr(), int32(numH)); err != nil {
		log.Fatalf("gelu failed: %v", err)
	}

	// 4. Layer 2 GEMM: tY = tH * tW2
	cfgGEMM2 := driver.LaunchConfig{
		GridDimX:  uint32((dOut + 15) / 16),
		GridDimY:  uint32((batch + 15) / 16),
		BlockDimX: 16,
		BlockDimY: 16,
	}
	if err := gemmFn.Launch(cfgGEMM2, tH.DevicePtr(), tW2.DevicePtr(), tY.DevicePtr(), int32(batch), int32(dOut), int32(dHidden)); err != nil {
		log.Fatalf("gemm2 failed: %v", err)
	}

	// 5. Bias add: tY += tb2
	cfgBias2 := driver.LaunchConfig{
		GridDimX:  uint32((dOut + 15) / 16),
		GridDimY:  uint32((batch + 15) / 16),
		BlockDimX: 16,
		BlockDimY: 16,
	}
	if err := biasFn.Launch(cfgBias2, tY.DevicePtr(), tb2.DevicePtr(), tY.DevicePtr(), int32(batch), int32(dOut)); err != nil {
		log.Fatalf("bias2 failed: %v", err)
	}

	// 6. Normalization: tOut = RMSNorm(tY, weight=tWeightNorm, eps=1e-5)
	if err := layernorm.ExecuteRMSNorm(ctx, nil, tY.DevicePtr(), tOut.DevicePtr(), tWeightNorm.DevicePtr(), batch, dOut, float32(1e-5)); err != nil {
		log.Fatalf("rmsnorm failed: %v", err)
	}

	if err := ctx.Synchronize(); err != nil {
		log.Fatalf("ctx.Synchronize failed: %v", err)
	}
	elapsed := time.Since(start)

	// Fetch result back to CPU
	gpuResult, err := tOut.ToCPUFloat32()
	if err != nil {
		log.Fatalf("ToCPUFloat32 failed: %v", err)
	}

	// Compute CPU ground truth
	refH := cpuMatMul(hX, hW1, batch, dHidden, dIn)
	cpuBiasAdd(refH, hb1, batch, dHidden)
	cpuGELU(refH)

	refY := cpuMatMul(refH, hW2, batch, dOut, dHidden)
	cpuBiasAdd(refY, hb2, batch, dOut)

	// Verify RMSNorm on CPU
	maxDiff := float64(0.0)
	for b := 0; b < batch; b++ {
		var sumSq float64
		for j := 0; j < dOut; j++ {
			val := float64(refY[b*dOut+j])
			sumSq += val * val
		}
		rms := math.Sqrt(sumSq/float64(dOut) + 1e-5)

		for j := 0; j < dOut; j++ {
			expected := float32((float64(refY[b*dOut+j]) / rms) * float64(hWeightNorm[j]))
			actual := gpuResult[b*dOut+j]
			diff := math.Abs(float64(actual - expected))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
	}

	fmt.Printf("GPU Forward Pass Latency: %v\n", elapsed)
	fmt.Printf("Max Absolute Difference vs CPU Reference: %.6e\n", maxDiff)
	if maxDiff < 1e-3 {
		fmt.Println("Verification: PASS (GPU output exactly matches CPU mathematical reference)")
	} else {
		log.Fatalf("Verification: FAIL (max difference %.6e exceeds tolerance)", maxDiff)
	}
	fmt.Println("================================================================================")
}
