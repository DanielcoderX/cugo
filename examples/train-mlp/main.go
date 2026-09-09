package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"runtime"
	"time"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/tensor"
)

func main() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	fmt.Println("================================================================================")
	fmt.Println(" cugo: Neural Network Training Loop with Autograd & AdamW (Zero CGO)")
	fmt.Println(" Task: 2-Layer MLP fitting non-linear function: Y = GELU(X*W1 + b1)*W2 + b2")
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
	fmt.Printf("Training Device: %s\n\n", devName)

	ctx, err := dev.CreateContext()
	if err != nil {
		log.Fatalf("CreateContext failed: %v", err)
	}
	defer ctx.Destroy()

	const batchSize = 32
	const inFeatures = 16
	const hiddenFeatures = 32
	const outFeatures = 1

	rng := rand.New(rand.NewSource(42))

	// 1. Synthetic training dataset: Y = sum(sin(X), dim=1)
	hX := make([]float32, batchSize*inFeatures)
	hY := make([]float32, batchSize*outFeatures)
	for i := range hX {
		hX[i] = rng.Float32()*2.0 - 1.0
	}
	for b := 0; b < batchSize; b++ {
		var sum float32
		for c := 0; c < inFeatures; c++ {
			sum += float32(math.Sin(float64(hX[b*inFeatures+c])))
		}
		hY[b] = sum
	}

	tX, err := tensor.NewFromFloat32(ctx, hX, batchSize, inFeatures)
	if err != nil { log.Fatal(err) }
	defer tX.Close()

	tY, err := tensor.NewFromFloat32(ctx, hY, batchSize, outFeatures)
	if err != nil { log.Fatal(err) }
	defer tY.Close()

	// 2. Trainable parameters with Xavier initialization
	hW1 := make([]float32, inFeatures*hiddenFeatures)
	hb1 := make([]float32, hiddenFeatures)
	hW2 := make([]float32, hiddenFeatures*outFeatures)
	hb2 := make([]float32, outFeatures)

	std1 := float32(math.Sqrt(2.0 / float64(inFeatures+hiddenFeatures)))
	for i := range hW1 { hW1[i] = (rng.Float32()*2.0 - 1.0) * std1 }

	std2 := float32(math.Sqrt(2.0 / float64(hiddenFeatures+outFeatures)))
	for i := range hW2 { hW2[i] = (rng.Float32()*2.0 - 1.0) * std2 }

	W1, _ := tensor.NewFromFloat32(ctx, hW1, inFeatures, hiddenFeatures); defer W1.Close()
	b1, _ := tensor.NewFromFloat32(ctx, hb1, hiddenFeatures); defer b1.Close()
	W2, _ := tensor.NewFromFloat32(ctx, hW2, hiddenFeatures, outFeatures); defer W2.Close()
	b2, _ := tensor.NewFromFloat32(ctx, hb2, outFeatures); defer b2.Close()

	W1.SetRequiresGrad(true)
	b1.SetRequiresGrad(true)
	W2.SetRequiresGrad(true)
	b2.SetRequiresGrad(true)

	params := []*tensor.Tensor{W1, b1, W2, b2}

	// 3. AdamW Optimizer
	opt, err := tensor.NewAdamW(params, 0.03, 0.9, 0.999, 1e-8, 0.001)
	if err != nil {
		log.Fatalf("NewAdamW failed: %v", err)
	}
	defer opt.Close()

	const epochs = 50
	var initialLoss, finalLoss float32

	fmt.Printf("Starting AdamW Training Loop for %d epochs...\n\n", epochs)
	start := time.Now()

	for epoch := 1; epoch <= epochs; epoch++ {
		// Zero gradients
		opt.ZeroGrad()

		// Forward Pass:
		// H1 = X * W1
		h1, err := tensor.MatMul(tX, W1); if err != nil { log.Fatal(err) }
		// H2 = H1 + b1
		h2, err := tensor.AddBias(h1, b1); if err != nil { log.Fatal(err) }
		// H3 = GELU(H2)
		h3, err := tensor.GELU(h2); if err != nil { log.Fatal(err) }
		// H4 = H3 * W2
		h4, err := tensor.MatMul(h3, W2); if err != nil { log.Fatal(err) }
		// Pred = H4 + b2
		pred, err := tensor.AddBias(h4, b2); if err != nil { log.Fatal(err) }

		// Compute MSE Loss
		loss, err := tensor.MSELoss(pred, tY)
		if err != nil {
			log.Fatal(err)
		}

		lossVal, _ := loss.ToCPUFloat32()
		curLoss := lossVal[0]
		if epoch == 1 {
			initialLoss = curLoss
		}
		if epoch == epochs {
			finalLoss = curLoss
		}

		if epoch%10 == 0 || epoch == 1 {
			fmt.Printf("Epoch [%2d/%2d]  |  MSE Loss: %.6f\n", epoch, epochs, curLoss)
		}

		// Backward Pass (Reverse-mode automatic differentiation)
		if err := loss.Backward(); err != nil {
			log.Fatalf("Backward failed at epoch %d: %v", epoch, err)
		}

		// Optimizer Step (AdamW parameter update on GPU)
		if err := opt.Step(); err != nil {
			log.Fatalf("opt.Step failed: %v", err)
		}

		// Clean up intermediate activations
		_ = loss.Close()
		_ = pred.Close()
		_ = h4.Close()
		_ = h3.Close()
		_ = h2.Close()
		_ = h1.Close()
	}

	trainDuration := time.Since(start)
	fmt.Printf("\nTraining Complete in %v (avg %.2f ms/epoch)\n", trainDuration, float64(trainDuration.Milliseconds())/float64(epochs))
	fmt.Printf("Initial Loss: %.6f  -->  Final Loss: %.6f (Reduction: %.1f%%)\n",
		initialLoss, finalLoss, (1.0-finalLoss/initialLoss)*100.0)

	if finalLoss < initialLoss*0.2 {
		fmt.Println("Verification: PASS (Autograd and AdamW successfully converged on GPU)")
	} else {
		log.Fatalf("Verification: FAIL (Loss did not converge adequately)")
	}
	fmt.Println("================================================================================")
}
