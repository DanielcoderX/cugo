package scale

import (
	_ "embed"
)

//go:generate nvcc -ptx -o scale.ptx scale.cu

// PTX contains embedded PTX bytecode for the scaleKernel with struct parameters.
//
//go:embed scale.ptx
var PTX []byte
