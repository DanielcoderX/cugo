package activation

import (
	_ "embed"
)

// PTX contains embedded PTX bytecode for neural network activation and bias kernels:
// - geluKernel(input, output, n)
// - reluKernel(input, output, n)
// - siluKernel(input, output, n)
// - sigmoidKernel(input, output, n)
// - biasAddKernel(input, bias, output, rows, cols)
//
//go:embed activation.ptx
var PTX []byte
