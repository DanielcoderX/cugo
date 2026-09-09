package autograd

import (
	_ "embed"
)

// PTX contains embedded PTX bytecode for autograd backward kernels:
// - geluBackwardKernel(grad_out, input, grad_in, n)
// - biasBackwardKernel(grad_out, grad_bias, rows, cols)
// - transposeKernel(in, out, rows, cols)
// - rmsnormBackwardKernel(grad_out, input, weight, grad_in, grad_weight, rows, cols, eps)
//
//go:embed backward.ptx
var PTX []byte
