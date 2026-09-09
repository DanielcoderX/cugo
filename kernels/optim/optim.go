package optim

import (
	_ "embed"
)

// PTX contains embedded PTX bytecode for optimizer update kernels:
// - adamwKernel(params, grads, exp_avg_m, exp_avg_v, lr, beta1, beta2, eps, weight_decay, bc1, bc2, n)
// - sgdKernel(params, grads, velocity, lr, momentum, weight_decay, n)
//
//go:embed optim.ptx
var PTX []byte
