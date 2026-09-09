package softmax_test

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/kernels/softmax"
)

func cpuSoftmax(input []float32, rows, cols int) []float32 {
	output := make([]float32, len(input))
	for r := 0; r < rows; r++ {
		rowStart := r * cols
		// Find max
		maxVal := float32(-1e38)
		for c := 0; c < cols; c++ {
			if input[rowStart+c] > maxVal {
				maxVal = input[rowStart+c]
			}
		}
		// Sum exp
		sumExp := float32(0)
		for c := 0; c < cols; c++ {
			val := float32(math.Exp(float64(input[rowStart+c] - maxVal)))
			output[rowStart+c] = val
			sumExp += val
		}
		// Normalize
		for c := 0; c < cols; c++ {
			output[rowStart+c] /= sumExp
		}
	}
	return output
}

func TestSoftmaxKernel(t *testing.T) {
	if err := driver.Init(); err != nil {
		t.Fatalf("driver.Init failed: %v", err)
	}

	count, err := driver.DeviceCount()
	if err != nil || count == 0 {
		t.Skip("no CUDA devices available")
	}

	dev, err := driver.GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice(0) failed: %v", err)
	}

	ctx, err := dev.CreateContext()
	if err != nil {
		t.Fatalf("CreateContext failed: %v", err)
	}
	defer ctx.Destroy()

	stream, err := ctx.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream failed: %v", err)
	}
	defer stream.Destroy()

	testCases := []struct {
		rows int
		cols int
	}{
		{rows: 1, cols: 64},
		{rows: 8, cols: 128},
		{rows: 64, cols: 512},
		{rows: 128, cols: 1024},
	}

	for _, tc := range testCases {
		totalElements := tc.rows * tc.cols
		totalBytes := totalElements * 4

		hInput := make([]float32, totalElements)
		for i := range hInput {
			hInput[i] = (rand.Float32() - 0.5) * 10.0
		}

		cpuExpected := cpuSoftmax(hInput, tc.rows, tc.cols)

		dInput, err := ctx.Alloc(uint64(totalBytes))
		if err != nil {
			t.Fatalf("Alloc dInput failed: %v", err)
		}
		defer ctx.Free(dInput)

		dOutput, err := ctx.Alloc(uint64(totalBytes))
		if err != nil {
			t.Fatalf("Alloc dOutput failed: %v", err)
		}
		defer ctx.Free(dOutput)


		inBytes := unsafe.Slice((*byte)(unsafe.Pointer(&hInput[0])), totalBytes)
		if err := ctx.CopyHtoD(dInput, inBytes); err != nil {
			t.Fatalf("CopyHtoD failed: %v", err)
		}

		if err := softmax.Execute(ctx, stream, dInput, dOutput, tc.rows, tc.cols); err != nil {
			t.Fatalf("softmax.Execute failed: %v", err)
		}

		if err := stream.Synchronize(); err != nil {
			t.Fatalf("stream.Synchronize failed: %v", err)
		}

		hOutput := make([]float32, totalElements)
		outBytes := unsafe.Slice((*byte)(unsafe.Pointer(&hOutput[0])), totalBytes)
		if err := ctx.CopyDtoH(outBytes, dOutput); err != nil {
			t.Fatalf("CopyDtoH failed: %v", err)
		}

		// Verify correctness
		for r := 0; r < tc.rows; r++ {
			rowSum := float32(0)
			for c := 0; c < tc.cols; c++ {
				idx := r*tc.cols + c
				rowSum += hOutput[idx]
				diff := math.Abs(float64(hOutput[idx] - cpuExpected[idx]))
				if diff > 1e-4 {
					t.Fatalf("Case [%dx%d] mismatch at row %d col %d: GPU=%f CPU=%f", tc.rows, tc.cols, r, c, hOutput[idx], cpuExpected[idx])
				}
			}
			if math.Abs(float64(rowSum-1.0)) > 1e-4 {
				t.Fatalf("Case [%dx%d] row %d sum != 1.0 (got %f)", tc.rows, tc.cols, r, rowSum)
			}
		}

		t.Logf("PASS: shape [%dx%d] (total %d floats)", tc.rows, tc.cols, totalElements)
	}
}
