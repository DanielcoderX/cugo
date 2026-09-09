package layernorm_test

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/DanielcoderX/cugo/driver"
	"github.com/DanielcoderX/cugo/kernels/layernorm"
)

func cpuRMSNorm(input []float32, weight []float32, rows, cols int, eps float32) []float32 {
	output := make([]float32, len(input))
	for r := 0; r < rows; r++ {
		rowStart := r * cols
		sumSq := float32(0)
		for c := 0; c < cols; c++ {
			val := input[rowStart+c]
			sumSq += val * val
		}
		meanSq := sumSq / float32(cols)
		rmsInv := float32(1.0 / math.Sqrt(float64(meanSq+eps)))

		for c := 0; c < cols; c++ {
			w := float32(1.0)
			if weight != nil {
				w = weight[c]
			}
			output[rowStart+c] = input[rowStart+c] * rmsInv * w
		}
	}
	return output
}

func TestRMSNormKernel(t *testing.T) {
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
		rows      int
		cols      int
		hasWeight bool
	}{
		{rows: 1, cols: 64, hasWeight: false},
		{rows: 8, cols: 256, hasWeight: true},
		{rows: 32, cols: 768, hasWeight: true},
		{rows: 64, cols: 1024, hasWeight: false},
	}

	const eps = float32(1e-5)

	for _, tc := range testCases {
		totalElements := tc.rows * tc.cols
		totalBytes := uint64(totalElements * 4)

		hInput := make([]float32, totalElements)
		for i := range hInput {
			hInput[i] = (rand.Float32() - 0.5) * 4.0
		}

		var hWeight []float32
		var dWeight driver.DevicePtr

		if tc.hasWeight {
			hWeight = make([]float32, tc.cols)
			for i := range hWeight {
				hWeight[i] = 0.5 + rand.Float32()
			}
			weightBytes := uint64(tc.cols * 4)
			var err error
			dWeight, err = ctx.Alloc(weightBytes)
			if err != nil {
				t.Fatalf("Alloc dWeight failed: %v", err)
			}
			defer ctx.Free(dWeight)

			wBytes := unsafe.Slice((*byte)(unsafe.Pointer(&hWeight[0])), weightBytes)
			if err := ctx.CopyHtoD(dWeight, wBytes); err != nil {
				t.Fatalf("CopyHtoD dWeight failed: %v", err)
			}
		}

		cpuExpected := cpuRMSNorm(hInput, hWeight, tc.rows, tc.cols, eps)

		dInput, err := ctx.Alloc(totalBytes)
		if err != nil {
			t.Fatalf("Alloc dInput failed: %v", err)
		}
		defer ctx.Free(dInput)

		dOutput, err := ctx.Alloc(totalBytes)
		if err != nil {
			t.Fatalf("Alloc dOutput failed: %v", err)
		}
		defer ctx.Free(dOutput)

		inBytes := unsafe.Slice((*byte)(unsafe.Pointer(&hInput[0])), totalBytes)
		if err := ctx.CopyHtoD(dInput, inBytes); err != nil {
			t.Fatalf("CopyHtoD dInput failed: %v", err)
		}

		if err := layernorm.ExecuteRMSNorm(ctx, stream, dInput, dOutput, dWeight, tc.rows, tc.cols, eps); err != nil {
			t.Fatalf("ExecuteRMSNorm failed: %v", err)
		}

		if err := stream.Synchronize(); err != nil {
			t.Fatalf("stream.Synchronize failed: %v", err)
		}

		hOutput := make([]float32, totalElements)
		outBytes := unsafe.Slice((*byte)(unsafe.Pointer(&hOutput[0])), totalBytes)
		if err := ctx.CopyDtoH(outBytes, dOutput); err != nil {
			t.Fatalf("CopyDtoH dOutput failed: %v", err)
		}

		// Verify correctness
		for i := 0; i < totalElements; i++ {
			diff := math.Abs(float64(hOutput[i] - cpuExpected[i]))
			if diff > 1e-4 {
				r := i / tc.cols
				c := i % tc.cols
				t.Fatalf("Case [%dx%d, weight=%v] mismatch at row %d col %d: GPU=%f CPU=%f",
					tc.rows, tc.cols, tc.hasWeight, r, c, hOutput[i], cpuExpected[i])
			}
		}

		t.Logf("PASS: shape [%dx%d, weight=%v] (%d elements)", tc.rows, tc.cols, tc.hasWeight, totalElements)
	}
}
